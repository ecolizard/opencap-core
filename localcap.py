import requests
import time
import json
import os
import shutil
from utilsServer import processTrial, runTestSession
import traceback
import logging
import glob
import numpy as np
from utilsAPI import getAPIURL, getWorkerType
from utilsAuth import getToken
from utils import getDataDirectory, checkTime, checkResourceUsage, sendStatusEmail
import os
import glob
import shutil
import requests
import json
import logging
import socket

from main import main
from utils import getDataDirectory
from utils import getTrialName
from utils import getTrialJson
from utils import downloadVideosFromServer
from utils import switchCalibrationForCamera
from utils import deleteCalibrationFiles
from utils import deleteStaticFiles
from utils import writeMediaToAPI
from utils import postCalibrationOptions
from utils import getCalibration
from utils import getModelAndMetadata
from utils import writeCalibrationOptionsToAPI
from utils import postMotionData
from utils import getNeutralTrialID
from utils import getCalibrationTrialID
from utils import sendStatusEmail
from utils import importMetadata
from utils import checkAndGetPosePickles
from utils import getTrialNameIdMapping
from utilsAuth import getToken
from utilsAPI import getAPIURL



def processLocalTrial(session_id, trial_id, trial_type='dynamic',
                 imageUpsampleFactor=4, poseDetector='OpenPose',
                 isDocker=True, resolutionPoseDetection='default',
                 bbox_thr=0.8, extrinsicTrialName='calibration',
                 deleteLocalFolder=True,
                 hasWritePermissions=True,
                 use_existing_pose_pickle=False,
                 batchProcess=False):
    # Get session directory
    session_name = session_id
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir, 'Data', session_name)
    trial_url = "{}{}{}/".format(API_URL, "trials/", trial_id)
    metadata_path = os.path.join(session_path, 'sessionMetadata.yaml')

    # Process the 3 different types of trials
    if trial_type == 'calibration':
        # delete extrinsic files if they exist.
        deleteCalibrationFiles(session_path)

        # download the videos
        trial_name = downloadVideosFromServer(session_id, trial_id, isDocker=isDocker,
                                              isCalibration=True, isStaticPose=False)

        # run calibration
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=True,
                 imageUpsampleFactor=imageUpsampleFactor, genericFolderNames=True)
        except Exception as e:
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = requests.patch(trial_url, data={"meta": json.dumps(error_msg)},
                               headers={"Authorization": "Token {}".format(API_TOKEN)})
            raise Exception('Calibration failed', e.args[0], e.args[1])

        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return

        # Write calibration images to django
        images_path = os.path.join(session_path, 'CalibrationImages')
        writeMediaToAPI(API_URL, images_path, trial_id, tag="calibration-img", deleteOldMedia=True)

        # Write calibration solutions to django
        writeCalibrationOptionsToAPI(session_path, session_id, calibration_id=trial_id,
                                     trialName=extrinsicTrialName)

    elif trial_type == 'static':
        # delete static files if they exist.
        deleteStaticFiles(session_path, staticTrialName='neutral')

        # Check for calibration to use on django, if not, check for switch calibrations and post result.
        calibrationOptions = getCalibration(session_id, session_path, trial_type=trial_type, getCalibrationOptions=True)

        # download the videos
        trial_name = downloadVideosFromServer(session_id, trial_id, isDocker=isDocker,
                                              isCalibration=False, isStaticPose=True)

        # Download the pose pickles to avoid re-running pose estimation.
        if batchProcess and use_existing_pose_pickle:
            checkAndGetPosePickles(trial_id, session_path, poseDetector, resolutionPoseDetection, bbox_thr)

        # If processTrial is run from app.py, poseDetector is set based on what
        # users select in the webapp, which is saved in metadata. Based on this,
        # we set resolutionPoseDetection or bbox_thr to the webapp defaults. If
        # processTrial is run from batchReprocess.py, then the settings used are
        # those passed as arguments to processTrial.
        if not batchProcess:
            sessionMetadata = importMetadata(metadata_path)
            poseDetector = sessionMetadata['posemodel']
            file_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(file_dir, 'defaultOpenCapSettings.json')) as f:
                defaultOpenCapSettings = json.load(f)
            if poseDetector.lower() == 'openpose':
                resolutionPoseDetection = defaultOpenCapSettings['openpose']
            elif poseDetector.lower() == 'hrnet':
                bbox_thr = defaultOpenCapSettings['hrnet']

                # run static
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=False,
                 poseDetector=poseDetector,
                 imageUpsampleFactor=imageUpsampleFactor,
                 scaleModel=True,
                 resolutionPoseDetection=resolutionPoseDetection,
                 genericFolderNames=True,
                 bbox_thr=bbox_thr,
                 calibrationOptions=calibrationOptions)
        except Exception as e:
            # Try to post pose pickles so can be used offline. This function will
            # error at kinematics most likely, but if pose estimation completed,
            # pickles will get posted
            try:
                # Write results to django
                if not batchProcess:
                    print('trial failed. posting pose pickles')
                    postMotionData(trial_id, session_path, trial_name=trial_name, isNeutral=True,
                                   poseDetector=poseDetector,
                                   resolutionPoseDetection=resolutionPoseDetection,
                                   bbox_thr=bbox_thr)
            except:
                pass

            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = requests.patch(trial_url, data={"meta": json.dumps(error_msg)},
                               headers={"Authorization": "Token {}".format(API_TOKEN)})
            raise Exception('Static trial failed', e.args[0], e.args[1])

        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return

        # Write videos to django
        video_path = getResultsPath(session_id, trial_id,
                                    resultType='neutralVideo', isDocker=isDocker)
        writeMediaToAPI(API_URL, video_path, trial_id, tag='video-sync', deleteOldMedia=True)

        # Write neutral pose images to django
        images_path = os.path.join(session_path, 'NeutralPoseImages')
        writeMediaToAPI(API_URL, images_path, trial_id, tag="neutral-img", deleteOldMedia=True)

        # Write visualizer jsons to django
        visualizerJson_path = getResultsPath(session_id, trial_id,
                                             resultType='visualizerJson',
                                             isDocker=isDocker)
        writeMediaToAPI(API_URL, visualizerJson_path, trial_id,
                        tag="visualizerTransforms-json", deleteOldMedia=True)

        # Write results to django
        postMotionData(trial_id, session_path, trial_name=trial_name, isNeutral=True,
                       poseDetector=poseDetector,
                       resolutionPoseDetection=resolutionPoseDetection,
                       bbox_thr=bbox_thr)

        # Write calibration options to django
        postCalibrationOptions(session_path, session_id, overwrite=True)

    elif trial_type == 'dynamic':
        # download calibration, model, and metadata if not existing
        getCalibration(session_id, session_path, trial_type=trial_type)
        getModelAndMetadata(session_id, session_path)

        # download the videos
        trial_name = downloadVideosFromServer(
            session_id, trial_id, isDocker=isDocker, isCalibration=False,
            isStaticPose=False)

        # Download the pose pickles to avoid re-running pose estimation.
        if batchProcess and use_existing_pose_pickle:
            checkAndGetPosePickles(trial_id, session_path, poseDetector, resolutionPoseDetection, bbox_thr)

        # If processTrial is run from app.py, poseDetector is set based on what
        # users select in the webapp, which is saved in metadata. Based on this,
        # we set resolutionPoseDetection or bbox_thr to the webapp defaults. If
        # processTrial is run from batchReprocess.py, then the settings used are
        # those passed as arguments to processTrial.
        if not batchProcess:
            sessionMetadata = importMetadata(metadata_path)
            poseDetector = sessionMetadata['posemodel']
            file_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(file_dir, 'defaultOpenCapSettings.json')) as f:
                defaultOpenCapSettings = json.load(f)
            if poseDetector.lower() == 'openpose':
                resolutionPoseDetection = defaultOpenCapSettings['openpose']
            elif poseDetector.lower() == 'hrnet':
                bbox_thr = defaultOpenCapSettings['hrnet']

                # run dynamic
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=False,
                 poseDetector=poseDetector,
                 imageUpsampleFactor=imageUpsampleFactor,
                 resolutionPoseDetection=resolutionPoseDetection,
                 genericFolderNames=True,
                 bbox_thr=bbox_thr)
        except Exception as e:
            # Try to post pose pickles so can be used offline. This function will
            # error at kinematics most likely, but if pose estimation completed,
            # pickles will get posted
            try:
                # Write results to django
                if not batchProcess:
                    print('trial failed. posting pose pickles')
                    postMotionData(trial_id, session_path, trial_name=trial_name, isNeutral=False,
                                   poseDetector=poseDetector,
                                   resolutionPoseDetection=resolutionPoseDetection,
                                   bbox_thr=bbox_thr)
            except:
                pass

            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = requests.patch(trial_url, data={"meta": json.dumps(error_msg)},
                               headers={"Authorization": "Token {}".format(API_TOKEN)})
            raise Exception('Dynamic trial failed.\n' + error_msg['error_msg_dev'], e.args[0], e.args[1])

        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return

        # Write videos to django
        video_path = getResultsPath(session_id, trial_id,
                                    resultType='sync_video', isDocker=isDocker)
        writeMediaToAPI(API_URL, video_path, trial_id, tag='video-sync', deleteOldMedia=True)

        # Write visualizer jsons to django
        visualizerJson_path = getResultsPath(session_id, trial_id,
                                             resultType='visualizerJson',
                                             isDocker=isDocker)
        writeMediaToAPI(API_URL, visualizerJson_path, trial_id,
                        tag="visualizerTransforms-json", deleteOldMedia=True)

        # Write results to django
        postMotionData(trial_id, session_path, trial_name=trial_name, isNeutral=False,
                       poseDetector=poseDetector,
                       resolutionPoseDetection=resolutionPoseDetection,
                       bbox_thr=bbox_thr)

    else:
        raise Exception('Wrong trial type. Options: calibration, static, dynamic.', 'TODO', 'TODO')

    # Remove data
    if deleteLocalFolder:
        shutil.rmtree(session_path)



logging.basicConfig(level=logging.INFO)

# if true, will delete entire data directory when finished with a trial
isDocker = False

# get start time
t = time.localtime()

# get session path from script arguments
if len(sys.argv) > 1:
    session_path = sys.argv[1]

# if session exists
if os.path.isdir(session_path):

    # check session is valid and videos are present for processing
    # @todo captures not found

    # dynamic, calibration, static
    trial_type = sys.argv[2]
    session_id = os.path.basename(session_path)

    try:
        processLocalTrial(trial["session"], trial["id"], trial_type=trial_type, isDocker=isDocker)

    except Exception as e:
        traceback.print_exc()
        args_as_strings = [str(arg) for arg in e.args]
        if len(args_as_strings) > 1 and 'pose detection timed out' in args_as_strings[1].lower():
            logging.info("Opencap failed.")
            message = "A backend OpenCap machine timed out during pose detection. It has been stopped."
            raise Exception('Worker failed. Stopped.')

    # Clean data directory
    if isDocker:
        folders = glob.glob(os.path.join(getDataDirectory(isDocker=True), 'Data', '*'))
        for f in folders:
            shutil.rmtree(f)
            logging.info('deleting ' + f)
