import sys

import requests
import time
import json
import os
import shutil

import yaml

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
from utils import getMetadataFromServer
from utilsAuth import getToken
from utilsAPI import getAPIURL

import pickle

# Note: This is modified from downloadVideosFromServer() in utils.py and will need to be updated if the original is updated.
def convertVideosToOpencap(session_path, session_id, trial_id, isDocker=False,
                          isCalibration=False, isStaticPose=False,
                          trial_name='calibration', session_name=None
                           ):
    if session_name is None:
        session_name = session_id

    trial_name = trial_name.replace(' ', '')

    trial = []

    # foreach .mpg file in the session path
    for video_path in glob.glob(os.path.join(session_path, '*.avi')):
        video_name = os.path.basename(video_path)
        video_name = video_name.replace(' ', '')
        video_name = video_name.replace('.avi', '')

        # create a video object
        video = {
            "video": video_path,
            "device_id": video_name,
            "parameters": {
                "model": "FLIR Blackfly S"
            }
        }
        trial.append(video)

    print("\nProcessing {}".format(trial_name))

    # The videos are not always organized in the same order. Here, we save
    # the order during the first trial processed in the session such that we
    # can use the same order for the other trials.
    if not os.path.exists(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle')):
        mappingCamDevice = {}
        for k, video in enumerate(trial["videos"]):
            os.makedirs(os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name),
                        exist_ok=True)
            video_path = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name,
                                      trial_id + ".mov")

            shutil.move(video["video"], video_path)

            mappingCamDevice[video["device_id"].replace('-', '').upper()] = k
        with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'wb') as handle:
            pickle.dump(mappingCamDevice, handle)
    else:
        with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'rb') as handle:
            mappingCamDevice = pickle.load(handle)
        for video in trial["videos"]:
            k = mappingCamDevice[video["device_id"].replace('-', '').upper()]
            videoDir = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name)
            os.makedirs(videoDir, exist_ok=True)
            video_path = os.path.join(videoDir, trial_id + ".mov")
            if not os.path.exists(video_path):
                if video['video']:
                    shutil.move(video["video"], video_path)

    # Import and save metadata
    sessionYamlPath = os.path.join(session_path, "sessionMetadata.yaml")
    if not os.path.exists(sessionYamlPath) or isStaticPose or isCalibration:
        if isCalibration:  # subject parameters won't be entered yet
            session_desc = getMetadataFromServer(session_id, justCheckerParams=isCalibration)
        else:  # subject parameters will be entered when capturing static pose
            session_desc = getMetadataFromServer(session_id)

        # Load iPhone models. @todo support FLIR
        phoneModel = []
        for i, video in enumerate(trial["videos"]):
            phoneModel.append(video['parameters']['model'])
        session_desc['iphoneModel'] = {'Cam' + str(i): phoneModel[i] for i in range(len(phoneModel))}

        # Save metadata.
        with open(sessionYamlPath, 'w') as file:
            yaml.dump(session_desc, file)

    return trial_name

''' Default session meta data:
calibrationSettings:
  overwriteDeployedIntrinsics: false
  saveSessionIntrinsics: false
checkerBoard:
  black2BlackCornersHeight_n: 8
  black2BlackCornersWidth_n: 11
  placement: ground
  squareSideLength_mm: 60
gender_mf: m
height_m: 1.89
iphoneModel:
  Cam0: iphone13,3
  Cam1: iphone13,3
  Cam2: iphone13,3
  Cam3: iphone13,3
  Cam4: iphone13,3
markerAugmentationSettings:
  markerAugmenterModel: LSTM
mass_kg: 83.2
openSimModel: LaiUhlrich2022
subjectID: defaultSubject'''

# Note: This is modified from processTrial() in utilsServer.py and will need to be updated if the original is updated.
def processLocalTrial(session_path, session_id, trial_id, trial_type='dynamic',
                      imageUpsampleFactor=4, poseDetector='OpenPose',
                      isDocker=True, resolutionPoseDetection='default',
                      bbox_thr=0.8, extrinsicTrialName='calibration',
                      deleteLocalFolder=True,
                      hasWritePermissions=True,
                      use_existing_pose_pickle=False,
                      batchProcess=False):
    # Get session directory
    session_name = session_id
    metadata_path = os.path.join(session_path, 'sessionMetadata.yaml')

    # Process the 3 different types of trials
    if trial_type == 'calibration':
        # delete extrinsic files if they exist.
        deleteCalibrationFiles(session_path)

        # download the videos
        trial_name = convertVideosToOpencap(session_path, session_id, trial_id, isDocker=isDocker,
                                           isCalibration=True, isStaticPose=False, trial_name='calibration')
        # run calibration
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=True,
                 imageUpsampleFactor=imageUpsampleFactor, genericFolderNames=True)
        except Exception as e:
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            raise Exception('Calibration failed', e.args[0], e.args[1])

        # Write calibration images to django
        images_path = os.path.join(session_path, 'CalibrationImages')

        # write locally to session path
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
        trial_name = convertVideosToOpencap(session_path, session_id, trial_id, isDocker=isDocker,
                                              isCalibration=False, isStaticPose=True, trial_name='neutral')

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
            raise Exception('Static trial failed', e.args[0], e.args[1])

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
            raise Exception('Dynamic trial failed.\n' + error_msg['error_msg_dev'], e.args[0], e.args[1])

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


####################################################################################################
# Local OpenCap processing
####################################################################################################
# This script processes a session on the local machine.
#
# Usage:
# python3 localcap.py /path/to/session/calibration calibration
# python3 localcap.py /path/to/session static
# python3 localcap.py /path/to/session dynamic
####################################################################################################

# get session path from script arguments
if len(sys.argv) > 1:
    session_path = sys.argv[1]

# if session exists
if os.path.isdir(session_path):

    # check session is valid and videos are present for processing
    # @todo captures not found

    # dynamic, calibration, static
    trial_type = sys.argv[2]

    # data/660d52880f89d3034501daf9/RecoveryInsights/patient_660d568bf69e2efee9ccc7f2/session_663a3091d4910ce33898342c/calibration
    session_id = os.path.basename(session_path)
    session_id = session_id.split('_')[1]

    patient_id = os.path.basename(os.path.dirname(session_path))
    patient_id = patient_id.split('_')[1]

    try:
        processLocalTrial(session_path, session_id, patient_id, trial_type=trial_type, isDocker=False)

    except Exception as e:
        traceback.print_exc()
        args_as_strings = [str(arg) for arg in e.args]
        if len(args_as_strings) > 1 and 'pose detection timed out' in args_as_strings[1].lower():
            logging.info("Opencap failed.")
            message = "A backend OpenCap machine timed out during pose detection. It has been stopped."
            raise Exception('processLocalTrial() failed.')



''' Opencap
    session = {
    "id": "6038ba9c-6d9d-4668-b215-6360001b92ce",
    "user": 3344,
    "public": false,
    "name": "6038ba9c",
    "qrcode": "https://mc-mocap-video-storage.s3.amazonaws.com/6038ba9c-6d9d-4668-b215-6360001b92ce.png?AWSAccessKeyId=AKIAZTRKQGHOES33JEGA&Signature=JdL80H%2FD39r3TL8gu3vLAnjNYZE%3D&Expires=1702995289",
    "meta": {
        "checkerboard": {
            "cols": "5",
            "rows": "4",
            "placement": "backWall",
            "square_size": "35"
        }
    },
    "trials": [
        {
            "id": "1e822abe-2851-4dfd-9027-26de32e47ec7",
            "session": "6038ba9c-6d9d-4668-b215-6360001b92ce",
            "name": "calibration",
            "status": "error",
            "videos": [],
            "results": [],
            "meta": {
                "error_msg": "No videos uploaded. Ensure phones are connected and you have stable internet connection.",
                "error_msg_dev": "No videos uploaded."
            },
            "created_at": "2023-12-19T10:56:58.296851Z",
            "updated_at": "2023-12-19T10:57:00.764437Z",
            "trashed": false,
            "trashed_at": null
        },
        {
            "id": "d77b8195-0f4b-4a65-9b4a-8c438d68282b",
            "session": "6038ba9c-6d9d-4668-b215-6360001b92ce",
            "name": "calibration",
            "status": "error",
            "videos": [],
            "results": [],
            "meta": {
                "error_msg": "No videos uploaded. Ensure phones are connected and you have stable internet connection.",
                "error_msg_dev": "No videos uploaded."
            },
            "created_at": "2023-12-19T10:57:06.089001Z",
            "updated_at": "2023-12-19T10:57:08.988775Z",
            "trashed": false,
            "trashed_at": null
        },
        {
            "id": "c0242b3e-681a-4032-83e9-e1f06fd539ae",
            "session": "6038ba9c-6d9d-4668-b215-6360001b92ce",
            "name": "calibration",
            "status": "error",
            "videos": [],
            "results": [],
            "meta": {
                "error_msg": "No videos uploaded. Ensure phones are connected and you have stable internet connection.",
                "error_msg_dev": "No videos uploaded."
            },
            "created_at": "2023-12-19T10:57:13.802275Z",
            "updated_at": "2023-12-19T10:57:16.268406Z",
            "trashed": false,
            "trashed_at": null
        }
    ],
    "server": "171.65.92.179",
    "subject": null,
    "created_at": "2023-12-19T10:48:51.077407Z",
    "updated_at": "2023-12-19T10:57:13.469787Z",
    "trashed": false,
    "trashed_at": null
}'''
