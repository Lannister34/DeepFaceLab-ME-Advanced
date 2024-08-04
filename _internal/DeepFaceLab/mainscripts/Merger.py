import math
import multiprocessing
import traceback
from pathlib import Path

import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from merger import FrameInfo, InteractiveMergerSubprocessor, MergerConfig


def main (model_class_name=None,
          saved_models_path=None,
          training_data_src_path=None,
          force_model_name=None,
          input_path=None,
          output_path=None,
          output_mask_path=None,
          aligned_path=None,
          pak_name=None,
          force_gpu_idxs=None,
          xseg_models_path=None,
          cpu_only=None,
          reduce_clutter=False):
    io.log_info ("Preparing the synthesizer. \r\n")

    try:
        if not input_path.exists():        # Check if the input path exists
            io.log_err('Input directory not found (default %WORKSPACE%\data_dst), make sure it exists!') # If it does not exist, output an error message
            return

        if not output_path.exists():       # Check if the output path exists
            output_path.mkdir(parents=True, exist_ok=True) # If it does not exist, create the output path

        if not output_mask_path.exists():  # Check if the output mask path exists
            output_mask_path.mkdir(parents=True, exist_ok=True) # If it doesn't exist, create an output mask path

        if not saved_models_path.exists(): # Check if the model save path exists
            io.log_err('The model directory was not found (default %WORKSPACE%\model), make sure it exists!') # If it does not exist, output an error message
            return

        # Initialize model
        import models                       # Import the model
        model = models.import_model(model_class_name)(is_training=False,  # Initialize the model
                                                      saved_models_path=saved_models_path,
                                                      force_gpu_idxs=force_gpu_idxs,
                                                      force_model_name=force_model_name,
                                                      cpu_only=cpu_only,
                                                      reduce_clutter=reduce_clutter)

        predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()  # Get the merge configuration

        # Preparing MP functions
        predictor_func = MPFunc(predictor_func)  # Prepare multiprocess functions

        run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0  # Determine if it's running on the CPU
        xseg_256_extract_func = MPClassFuncOnDemand(XSegNet, 'extract',  # XSeg keying function
                                                    name='XSeg',
                                                    resolution=256,
                                                    weights_file_root=xseg_models_path,
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        face_enhancer_func = MPClassFuncOnDemand(FaceEnhancer, 'enhance',
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        is_interactive = io.input_bool ("Using an interactive synthesizer?", True) if not io.is_colab() else False # Whether to use the interactive merger

        if not is_interactive:  # If it's not interactive
            cfg.ask_settings()  # Request configuration settings
            
        subprocess_count = io.input_int("Number of working threads?", max(8, multiprocessing.cpu_count()), 
                                        valid_range=[1, multiprocessing.cpu_count()], help_message="Specifies the number of threads to process. Low values may affect performance. High values may cause memory errors. The value cannot be greater than the number of CPU cores" )

        input_path_image_paths = pathex.get_image_paths(input_path)  # Get the path of the image under the input path


        if cfg.type == MergerConfig.TYPE_MASKED:  # If the configuration type is Mask Merge
            if not aligned_path.exists():  # Check if the Aligned directory exists
                io.log_err('Aligned directory not found, make sure it exists.')  # Aligned directory does not exist error message
                return

            packed_samples = None
            try:
                packed_samples = samplelib.PackedFaceset.load(aligned_path, pak_name=pak_name)  # Trying to load a packaged facial set
            except:
                io.log_err(f"Error loading samplelib.PackedFaceset.load {str(aligned_path)}, {traceback.format_exc()}")


            if packed_samples is not None:  # If the packed face set is successfully loaded
                io.log_info ("Use packaged facial sets.")  # Use log messages from packaged facial sets
                def generator():  # Define the generator function
                    for sample in io.progress_bar_generator( packed_samples, "Gather Aligned Information"):  # Progress bar generator
                        filepath = Path(sample.filename)  # File path
                        yield filepath, DFLIMG.load(filepath, loader_func=lambda x: sample.read_raw_file()  )  # Load DFLIMG
            else:
                def generator():  # Define alternate generator functions
                    for filepath in io.progress_bar_generator( pathex.get_image_paths(aligned_path), "Collecting Aligned Information"):  # Progress bar generator
                        filepath = Path(filepath)  # File path
                        yield filepath, DFLIMG.load(filepath)  # Load DFLIMG

            alignments = {}  # Initialize the Aligned dictionary
            multiple_faces_detected = False  # Multi-face detection markers

            for filepath, dflimg in generator():  # Traversal generator
                if dflimg is None or not dflimg.has_data():  # If DFLIMG is invalid or has no data
                    io.log_err (f"{filepath.name} Not a dfl image file")  # Error messages for non-DFL image files
                    continue

                source_filename = dflimg.get_source_filename()  # Get source file name
                if source_filename is None:  # If the source file name does not exist
                    continue

                source_filepath = Path(source_filename)  # Source file path
                source_filename_stem = source_filepath.stem  # Source document base name

                if source_filename_stem not in alignments.keys():  # If the base name is not in the Aligned dictionary
                    alignments[ source_filename_stem ] = []  # Initialize keys

                alignments_ar = alignments[ source_filename_stem ]  # Get the Aligned array
                alignments_ar.append ( (dflimg.get_source_landmarks(), filepath, source_filepath, dflimg ) )  # Add Aligned information

                if len(alignments_ar) > 1:  # If the length of the Aligned array is greater than 1
                    multiple_faces_detected = True  # Set the multi-face detection flag to true

            if multiple_faces_detected:  # If multiple faces are detected
                io.log_info ("")  # Output empty log messages
                io.log_info ("Warning: multiple faces detected. Each source file should correspond to only one Aligned file.")  # Output warning messages
                io.log_info ("")  # Output empty log messages

            for a_key in list(alignments.keys()):
                a_ar = alignments[a_key]
                if len(a_ar) > 1:
                    for _, filepath, source_filepath, _ in a_ar:  # Iterate over Aligned arrays
                        io.log_info (f"Alignment file {filepath.name} references  {source_filepath.name} ")
                    io.log_info ("")

                alignments[a_key] = [ [a[0], a[3]] for a in a_ar]

            if multiple_faces_detected:
                io.log_info ("It is highly recommended to process each face separately.")
                io.log_info ("Use 'recover original filename' to determine the exact duplicates.")
                io.log_info ("")



            # build frames maunally
            frames = []
            for p in input_path_image_paths:
                cur_path = Path(p)
                data = alignments.get(cur_path.stem, None)
                if data == None:
                    frame_info=FrameInfo(filepath=cur_path)
                    frame = InteractiveMergerSubprocessor.Frame(frame_info=frame_info)
                else:
                    landmarks_list = [d[0] for d in data]
                    dfl_images_list = [d[1] for d in data]
                    frame_info=FrameInfo(filepath=cur_path, landmarks_list=landmarks_list, dfl_images_list=dfl_images_list)
                    frame = InteractiveMergerSubprocessor.Frame(frame_info=frame_info)

                frames.append(frame)

            # frames = [ InteractiveMergerSubprocessor.Frame( frame_info=FrameInfo(filepath=Path(p),
            #                                                          # landmarks_list = alignments_orig.get(Path(p).stem, None)
            #                                                         )
            #                                   )
            #            for p in input_path_image_paths ]

            if multiple_faces_detected:
                io.log_info ("Warning: Multiple faces detected. Motion blur will not be used.")
                io.log_info ("")
            else:
                s = 256  # Setting the size
                local_pts = [ (s//2-1, s//2-1), (s//2-1,0) ] # Center and upper points
                frames_len = len(frames)  # Frame length
                for i in io.progress_bar_generator( range(len(frames)) , "Calculate the motion vector"):  # Progress bar generator
                    fi_prev = frames[max(0, i-1)].frame_info  # Get previous frame information
                    fi      = frames[i].frame_info  # Get current frame information
                    fi_next = frames[min(i+1, frames_len-1)].frame_info  # Get next frame information
                    if len(fi_prev.landmarks_list) == 0 or \
                       len(fi.landmarks_list) == 0 or \
                       len(fi_next.landmarks_list) == 0:
                            continue

                    mat_prev = LandmarksProcessor.get_transform_mat ( fi_prev.landmarks_list[0], s, face_type=FaceType.FULL)  # Get the transformation matrix of the previous frame
                    mat      = LandmarksProcessor.get_transform_mat ( fi.landmarks_list[0]     , s, face_type=FaceType.FULL)  # Get the current frame transformation matrix
                    mat_next = LandmarksProcessor.get_transform_mat ( fi_next.landmarks_list[0], s, face_type=FaceType.FULL)  # Get next frame transformation matrix

                    pts_prev = LandmarksProcessor.transform_points (local_pts, mat_prev, True)  # Convert the previous frame point
                    pts      = LandmarksProcessor.transform_points (local_pts, mat, True)  # Convert the current frame point
                    pts_next = LandmarksProcessor.transform_points (local_pts, mat_next, True)  # Convert the next frame point

                    prev_vector = pts[0]-pts_prev[0]  # forward vector (math.)
                    next_vector = pts_next[0]-pts[0]  # backward vector

                    motion_vector = pts_next[0] - pts_prev[0]  # motion vector
                    fi.motion_power = npla.norm(motion_vector)  # exercise intensity

                    motion_vector = motion_vector / fi.motion_power if fi.motion_power != 0 else np.array([0,0],dtype=np.float32)  # Normalized motion vectors

                    fi.motion_deg = -math.atan2(motion_vector[1],motion_vector[0])*180 / math.pi  # angle of motion


        if len(frames) == 0:  # If there are no frames to merge
            io.log_info ("There are no frames in the input catalog to merge.")  # output message
        else:
            if False:  # Reserved for possible conditional extensions
                pass
            else:
                InteractiveMergerSubprocessor (  # Create an instance of the interactive merge subprocessor and run
                            is_interactive         = is_interactive,  # Interactive or not
                            merger_session_filepath = model.get_strpath_storage_for_file('merger_session.dat'),  # Merge session file paths
                            predictor_func         = predictor_func,  # predictive function
                            predictor_input_shape  = predictor_input_shape,  # Predicting Input Shapes
                            face_enhancer_func     = face_enhancer_func,  # Facial Enhancement Functions
                            xseg_256_extract_func  = xseg_256_extract_func,  # XSeg Extraction Function
                            merger_config          = cfg,  # Merge Configuration
                            frames                 = frames,  # frame list
                            frames_root_path       = input_path,  # root path of a frame
                            output_path            = output_path,  # output path
                            output_mask_path       = output_mask_path,  # Output Mask Path
                            model_iter             = model.get_iter(),  # Model Iteration
                            subprocess_count       = subprocess_count,  # Number of subprocesses
                        ).run()

        model.finalize()  # finalized model

    except Exception as e:  # Catching exceptions
        print ( traceback.format_exc() )  # Print the exception stack


"""
elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
filesdata = []
for filepath in io.progress_bar_generator(input_path_image_paths, "Collecting info"):
    filepath = Path(filepath)

    dflimg = DFLIMG.x(filepath)
    if dflimg is None:
        io.log_err ("%s Not a DFL image file" % (filepath.name) )
        continue
    filesdata += [ ( FrameInfo(filepath=filepath, landmarks_list=[dflimg.get_landmarks()] ), dflimg.get_source_filename() ) ]

filesdata = sorted(filesdata, key=operator.itemgetter(1)) #sort by source_filename
frames = []
filesdata_len = len(filesdata)
for i in range(len(filesdata)):
    frame_info = filesdata[i][0]

    prev_temporal_frame_infos = []
    next_temporal_frame_infos = []

    for t in range (cfg.temporal_face_count):
        prev_frame_info = filesdata[ max(i -t, 0) ][0]
        next_frame_info = filesdata[ min(i +t, filesdata_len-1 )][0]

        prev_temporal_frame_infos.insert (0, prev_frame_info )
        next_temporal_frame_infos.append (   next_frame_info )

    frames.append ( InteractiveMergerSubprocessor.Frame(prev_temporal_frame_infos=prev_temporal_frame_infos,
                                                frame_info=frame_info,
                                                next_temporal_frame_infos=next_temporal_frame_infos) )
"""

#interpolate landmarks
#from facelib import LandmarksProcessor
#from facelib import FaceType
#a = sorted(alignments.keys())
#a_len = len(a)
#
#box_pts = 3
#box = np.ones(box_pts)/box_pts
#for i in range( a_len ):
#    if i >= box_pts and i <= a_len-box_pts-1:
#        af0 = alignments[ a[i] ][0] ##first face
#        m0 = LandmarksProcessor.get_transform_mat (af0, 256, face_type=FaceType.FULL)
#
#        points = []
#
#        for j in range(-box_pts, box_pts+1):
#            af = alignments[ a[i+j] ][0] ##first face
#            m = LandmarksProcessor.get_transform_mat (af, 256, face_type=FaceType.FULL)
#            p = LandmarksProcessor.transform_points (af, m)
#            points.append (p)
#
#        points = np.array(points)
#        points_len = len(points)
#        t_points = np.transpose(points, [1,0,2])
#
#        p1 = np.array ( [ int(np.convolve(x[:,0], box, mode='same')[points_len//2]) for x in t_points ] )
#        p2 = np.array ( [ int(np.convolve(x[:,1], box, mode='same')[points_len//2]) for x in t_points ] )
#
#        new_points = np.concatenate( [np.expand_dims(p1,-1),np.expand_dims(p2,-1)], -1 )
#
#        alignments[ a[i] ][0]  = LandmarksProcessor.transform_points (new_points, m0, True).astype(np.int32)
