import pickle
from pathlib import Path

import cv2

from DFLIMG import *
from facelib import LandmarksProcessor, FaceType
from core.interact import interact as io
from core import pathex
from core.cv2ex import *
from samplelib import PackedFaceset


def is_packed(input_path):
    if PackedFaceset.path_contains(input_path):
        io.log_info (f'\n{input_path} Contains the packaged faceset! Please unzip it first.\n')
        return True

def save_faceset_metadata_folder(input_path):
    # Convert the input path to a Path object
    input_path = Path(input_path)

    # Check if the input path is packed, if so return
    if is_packed(input_path): 
        return

    # Define the path to the metadata file
    metadata_filepath = input_path / 'meta.dat'

    # Record keeping metadata information
    io.log_info(f"Save metadata to the {str(metadata_filepath)}\r\n")

    # Initialize an empty dictionary to store metadata
    d = {}

    # Iterate over the image files in the input path
    for filepath in io.progress_bar_generator(pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)
        
        # Load DFLIMG objects from image files
        dflimg = DFLIMG.load(filepath)

        # Check if DFLIMG is valid and contains data
        if dflimg is None or not dflimg.has_data():
            io.log_info(f"{filepath} not a DFL image file")
            continue
            
        # Get metadata from DFLIMG and store it in a dictionary
        dfl_dict = dflimg.get_dict()
        d[filepath.name] = (dflimg.get_shape(), dfl_dict)

    try:
        # Write dictionary containing metadata to metadata file
        with open(metadata_filepath, "wb") as f:
            f.write(pickle.dumps(d))
    except:
        # Raise an exception if a file write fails
        raise Exception('unsalvageable %s' % (filename))

    # Record information about editing images
    io.log_info("Now you can edit the image.")
    io.log_info("!!!!! Keep the same file name in the folder...")
    io.log_info("You can change the size of the image, the restore process will shrink it back to the original size.")
    io.log_info("After that, please use the restore metadata.")

def restore_faceset_metadata_folder(input_path):
    # Convert the input path to a Path object
    input_path = Path(input_path)

    # Check if the input path is packed, if so return
    if is_packed(input_path):
        return

    # Define the path to the metadata file
    metadata_filepath = input_path / 'meta.dat'
    
    # Record information on recovery metadata
    io.log_info(f"Recover metadata from {str(metadata_filepath)}.\r\n")

    # Log error if metadata file does not exist
    if not metadata_filepath.exists():
        io.log_err(f"can't find {str(metadata_filepath)}.")

    try:
        # Read a dictionary containing metadata from a metadata file
        with open(metadata_filepath, "rb") as f:
            d = pickle.loads(f.read())
    except:
        # Raise an exception if the file read fails
        raise FileNotFoundError(filename)

    # Iterate over image files in the input path with a specific extension
    for filepath in io.progress_bar_generator(pathex.get_image_paths(input_path, image_extensions=['.jpg'], return_Path_class=True), "Processing"):
        # Get saved metadata for the current file
        saved_data = d.get(filepath.name, None)
        
        # Check if saved metadata exists in the current file
        if saved_data is None:
            io.log_info(f"{filepath} no saved metadata")
            continue
        
        # Extract shape and DFL dictionaries from saved metadata
        shape, dfl_dict = saved_data

        # Reading images with OpenCV
        img = cv2_imread(filepath)

        # Resize the image if the image shape doesn't match the saved shape
        if img.shape != shape:
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)

            # Save the resized image with the original file name
            cv2_imwrite(str(filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Check for file extensions and process accordingly
        if filepath.suffix == '.jpg':
            # Load the DFLJPG object and set the dictionary
            dflimg = DFLJPG.load(filepath)
            dflimg.set_dict(dfl_dict)
            dflimg.save()
        else:
            # Skip if file extension is not '.jpg'
            continue

    # Delete metadata files after processing is complete
    metadata_filepath.unlink()
    
    
def add_landmarks_debug_images(input_path):

    if is_packed(input_path) : return

    io.log_info ("Adding Markers to Debug Images...")

    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        img = cv2_imread(str(filepath))

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} not a DFL image file")
            continue
        
        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            face_type = FaceType.fromString ( dflimg.get_face_type() )
            
            if face_type == FaceType.MARK_ONLY:
                rect = dflimg.get_source_rect()
                LandmarksProcessor.draw_rect_landmarks(img, rect, face_landmarks, FaceType.FULL )
            else:
                LandmarksProcessor.draw_landmarks(img, face_landmarks, transparent_mask=True )
            
            
            
            output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

def recover_original_aligned_filename(input_path):

    if is_packed(input_path) : return

    io.log_info ("Restore the original aligned file name...")

    files = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} not a DFL image file")
            continue

        files += [ [filepath, None, dflimg.get_source_filename(), False] ]

    files_len = len(files)
    for i in io.progress_bar_generator( range(files_len), "Sorting" ):
        fp, _, sf, converted = files[i]

        if converted:
            continue

        sf_stem = Path(sf).stem

        files[i][1] = fp.parent / ( sf_stem + '_0' + fp.suffix )
        files[i][3] = True
        c = 1

        for j in range(i+1, files_len):
            fp_j, _, sf_j, converted_j = files[j]
            if converted_j:
                continue

            if sf_j == sf:
                files[j][1] = fp_j.parent / ( sf_stem + ('_%d' % (c)) + fp_j.suffix )
                files[j][3] = True
                c += 1

    for file in io.progress_bar_generator( files, "Renaming", leave=False ):
        fs, _, _, _ = file
        dst = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (dst)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

    for file in io.progress_bar_generator( files, "Renaming" ):
        fs, fd, _, _ = file
        fs = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (fd)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

def export_faceset_mask(input_dir):
    for filename in io.progress_bar_generator(pathex.get_image_paths (input_dir), "Processing"):
        filepath = Path(filename)

        if '_mask' in filepath.stem:
            continue

        mask_filepath = filepath.parent / (filepath.stem+'_mask'+filepath.suffix)

        dflimg = DFLJPG.load(filepath)

        H,W,C = dflimg.shape

        seg_ie_polys = dflimg.get_seg_ie_polys()

        if seg_ie_polys.has_polys():
            mask = np.zeros ((H,W,1), dtype=np.float32)
            seg_ie_polys.overlay_mask(mask)
        elif dflimg.has_xseg_mask():
            mask = dflimg.get_xseg_mask()
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
        else:
            raise Exception(f'no mask in file {filepath}')


        cv2_imwrite(mask_filepath, (mask*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 100] )
