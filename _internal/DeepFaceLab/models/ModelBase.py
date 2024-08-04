import colorsys
import inspect
import multiprocessing
import operator
import os
import pickle
import shutil
import time
import datetime
from pathlib import Path
import yaml
from jsonschema import validate, ValidationError
import numpy as np
from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from samplelib import SampleGeneratorBase
from prettytable import PrettyTable


class ModelBase(object):
    # Constructor to initialize various parameters of the model
    def __init__(
        self,
        is_training=False,
        is_exporting=False,
        saved_models_path=None,
        training_data_src_path=None,
        training_data_dst_path=None,
        pretraining_data_path=None,
        pretrained_model_path=None,
        src_pak_name=None,
        dst_pak_name=None,
        no_preview=False,
        force_model_name=None,
        force_gpu_idxs=None,
        cpu_only=False,
        debug=False,
        force_model_class_name=None,
        config_training_file=None,
        auto_gen_config=False,
        silent_start=False,
        reduce_clutter=False,
        **kwargs,
    ):
        self.is_training = is_training  # Is it in training mode
        self.is_exporting = is_exporting  # Whether in export mode
        self.saved_models_path = saved_models_path  # Path to save the model
        self.training_data_src_path = training_data_src_path  # Training data source path
        self.training_data_dst_path = training_data_dst_path  # Training data target path
        self.pretraining_data_path = pretraining_data_path  # Pre-training data paths
        self.pretrained_model_path = pretrained_model_path  # Pre-trained model paths
        self.src_pak_name = src_pak_name  # Source packet name
        self.dst_pak_name = dst_pak_name  # Target packet name
        self.config_training_file = config_training_file  # Training profiles
        self.auto_gen_config = auto_gen_config  # Whether the configuration is automatically generated
        self.config_file_path = None  # Configuration file path
        self.no_preview = no_preview  # Whether not to show preview
        self.debug = debug  # Whether in debug mode
        self.reset_training = False  # Whether to reset training
        self.reduce_clutter = reduce_clutter  # Whether to reduce clutter
        self.author_name = "Shennong Chinese characterization"

        # Initialize model class name and model name
        self.model_class_name = model_class_name = Path(
            inspect.getmodule(self).__file__
        ).parent.name.rsplit("_", 1)[1]
        # Automatically set model name based on input parameters or model file
        if force_model_class_name is None:
            if force_model_name is not None:
                self.model_name = force_model_name
            else:
                while True:
                    # Collect data files for all models
                    saved_models_names = []
                    for filepath in pathex.get_file_paths(saved_models_path):
                        filepath_name = filepath.name
                        if filepath_name.endswith(f"{model_class_name}_data.dat"):
                            saved_models_names += [
                                (
                                    filepath_name.split("_")[0],
                                    os.path.getmtime(filepath),
                                )
                            ]

                    # Sorting models by modification time
                    saved_models_names = sorted(
                        saved_models_names, key=operator.itemgetter(1), reverse=True
                    )
                    saved_models_names = [x[0] for x in saved_models_names]

                    if len(saved_models_names) != 0:
                        if silent_start:
                            self.model_name = saved_models_names[0]
                            io.log_info(f'Silent Start: Selected Models "{self.model_name}"')
                        else:
                            io.log_info("Select a saved model, or enter a name to create a new model.")
                            io.log_info("[r] : rename")
                            io.log_info("[d] : removing")
                            io.log_info("")
                            for i, model_name in enumerate(saved_models_names):
                                s = f"[{i}] : {model_name} "
                                if i == 0:
                                    s += "- Last conducted"
                                io.log_info(s)

                            inp = io.input_str(f"", "0", show_default_value=False)
                            model_idx = -1
                            try:
                                model_idx = np.clip(
                                    int(inp), 0, len(saved_models_names) - 1
                                )
                            except:
                                pass

                            if model_idx == -1:
                                if len(inp) == 1:
                                    is_rename = inp[0] == "r"
                                    is_delete = inp[0] == "d"

                                    if is_rename or is_delete:
                                        if len(saved_models_names) != 0:
                                            if is_rename:
                                                name = io.input_str(
                                                    f"Enter the name of the model you want to rename"
                                                )
                                            elif is_delete:
                                                name = io.input_str(
                                                    f"Enter the name of the model you want to delete"
                                                )

                                            if name in saved_models_names:
                                                if is_rename:
                                                    new_model_name = io.input_str(
                                                        f"Enter a new name for the model"
                                                    )

                                                for filepath in pathex.get_paths(
                                                    saved_models_path
                                                ):
                                                    filepath_name = filepath.name
                                                    try:
                                                        (
                                                            model_filename,
                                                            remain_filename,
                                                        ) = filepath_name.split("_", 1)
                                                    except ValueError:
                                                        # Logic for handling when a file name cannot be split correctly
                                                        print(
                                                            "Warning: There are other files in the model directory (e.g. zip archives)"
                                                        )
                                                        print(
                                                            "Illegal file name:", filepath_name
                                                        )
                                                        continue  # Skip the current loop and move on to the next file

                                                    if model_filename == name:
                                                        if is_rename:
                                                            new_filepath = (
                                                                filepath.parent
                                                                / (
                                                                    new_model_name
                                                                    + "_"
                                                                    + remain_filename
                                                                )
                                                            )
                                                            filepath.rename(
                                                                new_filepath
                                                            )
                                                        elif is_delete:
                                                            filepath.unlink()
                                        continue

                                self.model_name = inp
                            else:
                                self.model_name = saved_models_names[model_idx]

                    else:
                        self.model_name = io.input_str(
                            f"Saved model not found. Create a new model, enter a name", "new"
                        )
                        self.model_name = self.model_name.replace("_", " ")
                    break

            self.model_name = self.model_name + "_" + self.model_class_name
        else:
            self.model_name = force_model_class_name

        self.iter = 0  # Current number of iterations
        self.options = {}  # Model Options
        self.formatted_dictionary = {}  # Formatted Dictionary
        self.options_show_override = {}  # Options for overriding the display
        self.loss_history = []  # Loss history
        self.sample_for_preview = None  # Preview Sample
        self.choosed_gpu_indexes = None  # Selected GPU Index
        model_data = {}
        # True if the yaml configuration file exists
        self.config_file_exists = False
        # True if the user selects the Read from external or internal configuration file option
        self.read_from_conf = False
        config_error = False

        # Check if config_training_file mode is enabled
        if config_training_file is not None:
            if not Path(config_training_file).exists():
                # Log an error message if the configuration file does not exist
                io.log_err(f"{config_training_file} does not exist, no configuration is used!")
            else:
                # Setting the configuration file path
                self.config_file_path = Path(self.get_strpath_def_conf_file())
        elif self.auto_gen_config:
            # If auto-generated configuration is enabled, set the configuration file path
            self.config_file_path = Path(self.get_model_conf_path())

        # If the configuration file path exists
        if self.config_file_path is not None:
            # Ask the user if they want to read the training options from a file
            self.read_from_conf = (
                io.input_bool(
                    f"Are you reading the training options from a file?",
                    True,
                    "Read the options from the configuration file instead of asking for each option individually",
                )
                if not silent_start
                else True
            )

            # If the user decides to read from an external or internal configuration file
            if self.read_from_conf:
                # Try to read the dictionary from an external or internal yaml file, based on the value of auto_gen_config
                self.options = self.read_from_config_file(self.config_file_path)
                # If the option dictionary is empty, the options will be loaded from the dat file
                if self.options is None:
                    io.log_info(f"Configuration file validation error, please check your configuration")
                    config_error = True
                elif not self.options.keys():
                    io.log_info(f"Configuration file does not exist. A standard configuration file will be created.")
                else:
                    io.log_info(f"Use the configuration file from {self.config_file_path}.")
                    self.config_file_exists = True

        # Set the path to the model data file
        self.model_data_path = Path(self.get_strpath_storage_for_file("data.dat"))
        # If the model data file exists
        if self.model_data_path.exists():
            io.log_info(f"Load model {self.model_name}...")
            # Reading and parsing model data from a file
            model_data = pickle.loads(self.model_data_path.read_bytes())
            self.iter = model_data.get("iter", 0)
            try:
                self.author_name = model_data.get("author_name", "Shennong Chinese characterization")
            except KeyError:
                # If the fetch fails, ignore the error and set the attribute to "Shennong Chinese"
                print(f"Failed to read author name, model was upgraded from old DFL version")
                self.author_name = "Shennong Chinese characterization"

            # If the number of iterations is not 0, it means that the model is not running for the first time
            if self.iter != 0:
                # If the user chooses not to read the options from the yaml file, the options are read from the .dat file
                if not self.config_file_exists:
                    self.options = model_data["options"]
                # Read loss history, preview samples and selected GPU indexes from model data
                self.loss_history = model_data.get("loss_history", [])
                self.sample_for_preview = model_data.get("sample_for_preview", None)
                self.choosed_gpu_indexes = model_data.get("choosed_gpu_indexes", None)

        # If it is the first run of the model
        if self.is_first_run():
            io.log_info("\n model is run for the first time.")

        # If it's a silent boot
        if silent_start:
            # Setting Up Device Configuration
            if force_gpu_idxs is not None:
                self.device_config = (
                    nn.DeviceConfig.GPUIndexes(force_gpu_idxs)
                    if not cpu_only
                    else nn.DeviceConfig.CPU()
                )
                io.log_info(
                    f"Silent boot: selected devices {'s' if len(force_gpu_idxs) > 0 else ''} {'CPU' if self.device_config.cpu_only else [device.name for device in self.device_config.devices]}"
                )
            else:
                self.device_config = nn.DeviceConfig.BestGPU()
                io.log_info(
                    f"Silent boot: selected devices {'CPU' if self.device_config.cpu_only else self.device_config.devices[0].name}"
                )
        else:
            # Setting the device configuration in non-silent boot
            self.device_config = (
                nn.DeviceConfig.GPUIndexes(
                    force_gpu_idxs
                    or nn.ask_choose_device_idxs(suggest_best_multi_gpu=True)
                )
                if not cpu_only
                else nn.DeviceConfig.CPU()
            )

        # Initialize the neural network
        nn.initialize(self.device_config)

        ####
        # Setting the path to the default options file
        self.default_options_path = (
            saved_models_path / f"{self.model_class_name}_default_options.dat"
        )
        self.default_options = {}
        # If the default options file exists
        if self.default_options_path.exists():
            try:
                # Read and parse default options from a file
                self.default_options = pickle.loads(
                    self.default_options_path.read_bytes()
                )
            except:
                pass

        # Initialize preview history selection and batch size
        self.choose_preview_history = False
        self.batch_size = self.load_or_def_option("batch_size", 1)
        #####

        # Skip all pending inputs
        io.input_skip_pending()
        # Initialization Options
        self.on_initialize_options()

        # If it is the first run of the model
        if self.is_first_run():
            # Save the current options as default only the first time the model is run
            self.default_options_path.write_bytes(pickle.dumps(self.options))

        # Save Configuration File
        if (
            self.read_from_conf
            and not self.config_file_exists
            and not config_error
            and not self.config_file_path is None
        ):
            self.save_config_file(self.config_file_path)

        # Get various configuration parameters from the options
        # self.author_name = self.options.get("author_name", "")
        self.autobackup_hour = self.options.get("autobackup_hour", 0)
        self.maximum_n_backups = self.options.get("maximum_n_backups", 24)
        self.write_preview_history = self.options.get("write_preview_history", False)
        self.target_iter = self.options.get("target_iter", 0)
        self.random_flip = self.options.get("random_flip", True)
        self.random_src_flip = self.options.get("random_src_flip", False)
        self.random_dst_flip = self.options.get("random_dst_flip", True)
        self.retraining_samples = self.options.get("retraining_samples", False)
        
        if self.model_class_name =="ME":
            self.super_warp = self.options.get("super_warp", False)
            if self.options["super_warp"] == True:
                self.rotation_range=[-12,12]
                self.scale_range=[-0.2, 0.2]
        # Completion of initialization
        self.on_initialize()
        # Batch size in update options
        self.options["batch_size"] = self.batch_size

        self.preview_history_writer = None
        # If in training mode
        if self.is_training:
            # Setting the path for preview history and automatic backups
            self.preview_history_path = self.saved_models_path / (
                f"{self.get_model_name()}_history"
            )
            self.autobackups_path = self.saved_models_path / (
                f"{self.get_model_name()}_autobackups"
            )

            # If write preview history is enabled or in a Colab environment
            if self.write_preview_history or io.is_colab():
                if not self.preview_history_path.exists():
                    self.preview_history_path.mkdir(exist_ok=True)
                else:
                    # If the number of iterations is 0, clean up the preview history folder
                    if self.iter == 0:
                        for filename in pathex.get_image_paths(
                            self.preview_history_path
                        ):
                            Path(filename).unlink()

            # Check if the training data generator is set
            if self.generator_list is None:
                raise ValueError("You didnt set_training_data_generators()")
            else:
                for i, generator in enumerate(self.generator_list):
                    if not isinstance(generator, SampleGeneratorBase):
                        raise ValueError(
                            "training data generator is not subclass of SampleGeneratorBase"
                        )

            # Update Preview Sample
            self.update_sample_for_preview(
                choose_preview_history=self.choose_preview_history
            )

            # If an automatic backup time is set
            if self.autobackup_hour != 0:
                self.autobackup_start_time = time.time()

                if not self.autobackups_path.exists():
                    self.autobackups_path.mkdir(exist_ok=True)

        # Prints the training summary, with an if because it has already been printed once before, and this is the modification that needs to be reproduced.
        if self.ask_override:
            io.log_info( self.get_summary_text() )
    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        # Update Preview Sample
        if self.sample_for_preview is None or choose_preview_history or force_new:
            # If Preview History is selected and you are in a Windows environment
            if choose_preview_history and io.is_support_windows():
                wnd_name = (
                    "[p] - next. [space] - switch preview type. [enter] - confirm."
                )
                io.log_info(f"Select the preview image to evolve. {wnd_name}")
                io.named_window(wnd_name)
                io.capture_keys(wnd_name)
                choosed = False
                preview_id_counter = 0
                mask_changed = False
                while not choosed:
                    if not mask_changed:
                        self.sample_for_preview = self.generate_next_samples()
                        previews = self.get_history_previews()

                    io.show_image(
                        wnd_name,
                        (previews[preview_id_counter % len(previews)][1] * 255).astype(
                            np.uint8
                        ),
                    )

                    while True:
                        key_events = io.get_key_events(wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = (
                            key_events[-1]
                            if len(key_events) > 0
                            else (0, 0, False, False, False)
                        )
                        if key == ord("\n") or key == ord("\r"):
                            choosed = True
                            break
                        elif key == ord(" "):
                            preview_id_counter += 1
                            mask_changed = True
                            break
                        elif key == ord("p"):
                            if mask_changed:
                                mask_changed = False
                            break

                        try:
                            io.process_messages(0.1)
                        except KeyboardInterrupt:
                            choosed = True

                io.destroy_window(wnd_name)
            else:
                # Generate the next sample as a preview
                self.sample_for_preview = self.generate_next_samples()

        try:
            self.get_history_previews()
        except:
            self.sample_for_preview = self.generate_next_samples()

        self.last_sample = self.sample_for_preview

    def load_or_def_option(self, name, def_value):
        # Loads the value with the specified name from options, tries to load it from default_options if it doesn't exist, and ends up using the default value
        options_val = self.options.get(name, None)
        if options_val is not None:
            return options_val

        def_opt_val = self.default_options.get(name, None)
        if def_opt_val is not None:
            return def_opt_val

        return def_value

    def load_inter_dims(self):
        # Tries to load the inter_dims value from options, returns False if it does not exist
        try:
            v = self.options["inter_dims"]
        except KeyError:
            return False
        return v

    def ask_override(self):
        # Printed Model Summary
        io.log_info( self.get_summary_text() )
        # Set the delay time, 5 seconds if in a Colab environment, 10 seconds otherwise
        time_delay = 5 if io.is_colab() else 10
        # If in training state and not the first run, ask the user whether to override the model settings
        return (
            self.is_training
            and not self.is_first_run()
            and io.input_in_time(
                f"Press Enter within {time_delay} seconds to modify the model settings.", time_delay
            )
        )

    def ask_reset_training(self):
        # Ask the user if they want to reset the iteration counters and loss charts
        self.reset_training = io.input_bool(
            "Do you want to reset the iteration counter and loss chart?",
            False,
            "Resets the model's iteration counters and loss charts, but your model does not lose training progress."
            "This can be useful if you always use the same model for multiple training sessions.",
        )

        if self.reset_training:
            self.set_iter(0)
                
    def ask_author_name(self, default_value="Crucified Midget :^)"):

        # Ask the user to enter a name
        self.author_name = io.input_str(
            "Model author Author name",
            default_value,
            help_message="Author's signature displayed",
        )

    def ask_autobackup_hour(self, default_value=0):
        # Load the autobackup_hour option and use the default if it does not exist
        default_autobackup_hour = self.options["autobackup_hour"] = (
            self.load_or_def_option("autobackup hour", default_value)
        )
        # Ask the user to enter a time interval for automatic backups
        self.options["autobackup_hour"] = io.input_int(
            f"Autobackup hour",
            default_autobackup_hour,
            add_info="0..24",
            help_message="Model files and previews are automatically backed up every N hours. The latest backups are located in the last folder of model/<>_autobackups in ascending order of name",
        )

    def ask_maximum_n_backups(self, default_value=24):
        # Load the maximum_n_backups option and use the default if it does not exist
        default_maximum_n_backups = self.options["maximum_n_backups"] = (
            self.load_or_def_option("maximum_n_backups", default_value)
        )
        # Ask the user to enter the maximum number of backups
        self.options["maximum_n_backups"] = io.input_int(
            f"Maximum n backups",
            default_maximum_n_backups,
            help_message="The maximum number of backups located in model/<>_autobackups. Entering 0 will allow it to do any number of autobackups based on the number of occurrences.",
        )

    def ask_write_preview_history(self, default_value=False):
        # Load the write_preview_history option, or use the default if it doesn't exist
        default_write_preview_history = self.load_or_def_option(
            "write_preview_history", default_value
        )
        # Ask the user whether to write the preview history
        self.options["write_preview_history"] = io.input_bool(
            f"Write preview history",
            default_write_preview_history,
            help_message="The preview graph evolution history will be written to the <ModelName>_history folder.",
        )

        if self.options["write_preview_history"]:
            if io.is_support_windows():
                self.choose_preview_history = io.input_bool(
                    "Select the serial number of the image for which you want to record a preview", False
                )
            elif io.is_colab():
                self.choose_preview_history = io.input_bool(
                    "Randomly select the serial number of the image to record a preview of",
                    False,
                    help_message="If you reuse the same model on a different character, the preview image evolution history will record the old face. Select No unless you want to change the source src/target dst to a new character!",
                )

    def ask_target_iter(self, default_value=0):
        # Load the target_iter option, or use the default if it doesn't exist
        default_target_iter = self.load_or_def_option("target_iter", default_value)
        # Ask the user to enter a target number of iterations
        self.options["target_iter"] = max(
            0, io.input_int("Target iter", default_target_iter)
        )

    def ask_random_flip(self):
        # Load the random_flip option, or use the default if it doesn't exist
        default_random_flip = self.load_or_def_option("random_flip", True)
        # Ask the user if they randomly flip their face
        self.options["random_flip"] = io.input_bool(
            "Random flip",
            default_random_flip,
            help_message="The predicted faces look more natural, but the src face set should cover all the same orientations as the dst face set.",
        )

    def ask_random_src_flip(self):
        # Load the random_src_flip option, or use the default if it doesn't exist
        default_random_src_flip = self.load_or_def_option("random_src_flip", False)
        # Ask the user if they randomly flip the SRC face
        self.options["random_src_flip"] = io.input_bool(
            "Random src flip",
            default_random_src_flip,
            help_message="Randomly flips the SRC face set horizontally. Covers more angles, but faces may look less natural.",
        )

    def ask_random_dst_flip(self):
        # Load the random_dst_flip option and use the default if it does not exist
        default_random_dst_flip = self.load_or_def_option("random_dst_flip", True)
        # Ask the user if they randomly flip the DST face
        self.options["random_dst_flip"] = io.input_bool(
            "Random dst flip",
            default_random_dst_flip,
            help_message="Randomize the set of DST faces by flipping them horizontally. Makes src->dst generalize better if src random flip is not enabled.",
        )

    def ask_batch_size(self, suggest_batch_size=None, range=None):
        # Load the batch_size option and use the recommended value or the current batch size if it does not exist
        default_batch_size = self.load_or_def_option(
            "batch_size", suggest_batch_size or self.batch_size
        )
        # Ask the user to enter the batch size
        batch_size = max(
            0,
            io.input_int(
                "Batch size",
                default_batch_size,
                valid_range=range,
                help_message="Larger batch sizes help generalize the neural network, but may result in memory overflow errors. Please adjust it manually to suit your graphics card.",
            ),
        )

        if range is not None:
            batch_size = np.clip(batch_size, range[0], range[1])

        self.options["batch_size"] = self.batch_size = batch_size

    def ask_retraining_samples(self, default_value=False):
        # Load the retraining_samples option, or use the default if it doesn't exist
        default_retraining_samples = self.load_or_def_option(
            "retraining_samples", default_value
        )
        # Ask the user whether to retrain high loss samples
        self.options["retraining_samples"] = io.input_bool(
            "Retraining samples",
            default_retraining_samples,
            help_message="Periodically retrain the last 16 high loss samples",
        )
    
    def ask_quick_opt(self):
        # Load the random_src_flip option, or use the default if it doesn't exist
        default_quick_opt = self.load_or_def_option("quick_opt", False)
        # Ask the user if they randomly flip the SRC face
        self.options["quick_opt"] = io.input_bool(
            "train eye_mouth (y) train skin (n)",
            default_quick_opt,
            help_message="Train the eyes and mouth first, and wait to train the skin each time the change in saved loss is less than 0.01 (GAN)",
        )
        
    # Methods that can be overridden
    def on_initialize_options(self):
        pass

    # Methods that can be overridden
    def on_initialize(self):
        """
        Initialize your model

        Storing and retrieving your model options in self.options['']

        See the example
        """
        pass

    # Methods that can be overridden
    def onSave(self):
        # Save your model here
        pass

    # Methods that can be overridden
    def onTrainOneIter(self, sample, generator_list):
        # Train your model here

        # Return loss array
        return (("loss_src", 0), ("loss_dst", 0))

    # Methods that can be overridden
    def onGetPreview(self, sample, for_history=False, filenames=None):
        # You can return multiple previews
        # Returns [('preview_name', preview_rgb), ...]
        return []

    # Methods that can be overridden, if you want the model name to be different from the folder name
    def get_model_name(self):
        return self.model_name

    # Method that can be overridden to return [[model, filename], ...] List
    def get_model_filename_list(self):
        return []

    # Methods that can be overridden
    def get_MergerConfig(self):
        # Returns predictor_func, predictor_input_shape and MergerConfig() for the model.
        raise NotImplementedError

    # Methods that can be overridden
    def get_config_schema_path(self):
        raise NotImplementedError

    # Methods that can be overridden
    def get_formatted_configuration_path(self):
        return "None"
        # raise NotImplementedError

    def get_pretraining_data_path(self):
        # Path to return pre-trained data
        return self.pretraining_data_path

    def get_target_iter(self):
        # Return the number of target iterations
        return self.target_iter

    def is_reached_iter_goal(self):
        # Check that the target number of iterations has been reached
        return self.target_iter != 0 and self.iter >= self.target_iter

    def get_previews(self):
        # Get preview image
        return self.onGetPreview(self.last_sample, filenames=self.last_sample_filenames)

    def get_static_previews(self):
        # Get static preview image
        return self.onGetPreview(self.sample_for_preview)

    def get_history_previews(self):
        # Get history preview image
        return self.onGetPreview(self.sample_for_preview, for_history=True)

    def get_preview_history_writer(self):
        # Get or create a preview history writer
        if self.preview_history_writer is None:
            self.preview_history_writer = PreviewHistoryWriter()
        return self.preview_history_writer

    def save(self):
        # Keeping a summary of the model
        Path(self.get_summary_path()).write_text(
            self.get_summary_text(), encoding="utf-8"
        )

        # Call the function that saves the model
        self.onSave()

        # If automatic configuration generation is enabled
        if self.auto_gen_config:
            path = Path(self.get_model_conf_path())
            self.save_config_file(path)

        # Model data ready to be saved
        model_data = {
            "iter": self.iter,
            "author_name": self.author_name,
            "options": self.options,
            "loss_history": self.loss_history,
            "sample_for_preview": self.sample_for_preview,
            "choosed_gpu_indexes": self.choosed_gpu_indexes,
        }

        # Creating a temporary file path
        temp_model_data_path = Path(self.model_data_path).with_suffix(".tmp")

        # Write serialized model data to a temporary file
        with open(temp_model_data_path, "wb") as f:
            pickle.dump(model_data, f)

        # Using write_bytes_safe to move temporary files to their final destination
        pathex.write_bytes_safe(Path(self.model_data_path), temp_model_data_path)

        # If an automatic backup time is set
        if self.autobackup_hour != 0:
            diff_hour = int((time.time() - self.autobackup_start_time) // 3600)

            if diff_hour > 0 and diff_hour % self.autobackup_hour == 0:
                self.autobackup_start_time += self.autobackup_hour * 3600
                self.create_backup()

    def __convert_type_write(self, value):
        # Converting data types for writing
        if isinstance(value, (np.int32, np.float64, np.int64)):
            return value.item()
        else:
            return value

    def __update_nested_dict(self, nested_dict, key, val):
        # Updating keys in a nested dictionary
        if key in nested_dict:
            nested_dict[key] = self.__convert_type_write(val)
            return True
        for v in nested_dict.values():
            if isinstance(v, dict):
                if self.__update_nested_dict(v, key, val):
                    return True
        return False

    def __iterate_read_dict(self, nested_dict, new_dict=None):
        # Iterative reading of nested dictionaries
        if new_dict is None:
            new_dict = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                new_dict.update(self.__iterate_read_dict(v, new_dict))
            else:
                new_dict[k] = v
        return new_dict

    def read_from_config_file(self, filepath, keep_nested=False, validation=True):
        """
        Read options from the configuration yaml file.

        Parameters.
            filepath (str|Path): The path to read the configuration file.
            keep_nested (bool, optional): if false, keep the dictionary nested, otherwise not. Default is False.
            validation (bool, optional): If true, validate the dictionary. Defaults to True.

        Returns.
            [dict]: optional dictionary.
        """
        data = {}
        try:
            # Open the file and read the data
            with open(filepath, "r", encoding="utf-8") as file, open(
                self.get_config_schema_path(), "r"
            ) as schema:
                data = yaml.safe_load(file)
                if not keep_nested:
                    data = self.__iterate_read_dict(data)
                if validation:
                    validate(data, yaml.safe_load(schema))
        except FileNotFoundError:
            return {}
        except ValidationError as ve:
            io.log_err(f"{ve}")
            return None

        return data

    def save_config_file(self, filepath):
        """
        Save options to configuration yaml file

        Parameters.
            filepath (str|Path): Path to save the configuration file.
        """
        formatted_dict = self.read_from_config_file(
            self.get_formatted_configuration_path(), keep_nested=True, validation=False
        )

        # Update dictionary and save
        for key, value in self.options.items():
            if not self.__update_nested_dict(formatted_dict, key, value):
                pass
                # print(f"'{key}' not saved in config file")

        try:
            with open(filepath, "w", encoding="utf-8") as file:
                yaml.dump(
                    formatted_dict,
                    file,
                    Dumper=yaml.SafeDumper,
                    allow_unicode=True,
                    default_flow_style=False,
                    encoding="utf-8",
                    sort_keys=False,
                )
        except OSError as exception:
            io.log_info("Unable to write YAML configuration file -> ", exception)

    def create_backup(self):
        io.log_info("Backup being created...", end="\r")

        # Make sure the backup path exists
        if not self.autobackups_path.exists():
            self.autobackups_path.mkdir(exist_ok=True)

        # Prepare a list of backup files
        bckp_filename_list = [
            self.get_strpath_storage_for_file(filename)
            for _, filename in self.get_model_filename_list()
        ]
        bckp_filename_list += [str(self.get_summary_path()), str(self.model_data_path)]

        # Creating a new backup
        idx_str = (
            datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S_")
            + str(self.iter // 1000)
            + "k"
        )
        idx_backup_path = self.autobackups_path / idx_str
        idx_backup_path.mkdir()
        for filename in bckp_filename_list:
            shutil.copy(str(filename), str(idx_backup_path / Path(filename).name))
        # Generate a preview image and save it in a new backup
        previews = self.get_previews()
        plist = []
        for i in range(len(previews)):
            name, bgr = previews[i]
            plist += [(bgr, idx_backup_path / (("preview_%s.jpg") % (name)))]

        if len(plist) != 0:
            self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

        # Check if the maximum number of backups is exceeded
        if self.maximum_n_backups != 0:
            all_backups = sorted(
                [x for x in self.autobackups_path.iterdir() if x.is_dir()]
            )
            while len(all_backups) > self.maximum_n_backups:
                oldest_backup = all_backups.pop(0)
                pathex.delete_all_files(oldest_backup)
                oldest_backup.rmdir()

    def debug_one_iter(self):
        # Debug one iteration to stack the generated images into a square image
        images = []
        for generator in self.generator_list:
            for i, batch in enumerate(next(generator)):
                if len(batch.shape) == 4:
                    images.append(batch[0])

        return imagelib.equalize_and_stack_square(images)

    def generate_next_samples(self):
        # Generate the next batch of samples

        sample = []  # For storing sample data
        sample_filenames = []  # Used to store sample file names

        # Iterate through the list of generators
        for generator in self.generator_list:
            # Check if the generator is initialized
            if generator.is_initialized():
                # Generate the next batch of samples
                batch = generator.generate_next()
                # If samples are generated in tuple form (containing data and filenames)
                if type(batch) is tuple:
                    sample.append(batch[0])  # Adding sample data to the sample list
                    sample_filenames.append(batch[1])  # Add sample filenames to the filename list
                else:
                    sample.append(batch)  # Adding samples to the sample list
            else:
                sample.append([])  # If the generator is not initialized, add the empty list to the sample list

        # Updating the last batch of samples and file names
        self.last_sample = sample  # Updating of data from the previous sample batch
        self.last_sample_filenames = sample_filenames  # Update the last batch of sample file names

        # Returns the generated sample
        return sample

    # Methods that can be overridden
    def should_save_preview_history(self):
        # Determine if the preview history should be saved
        return (not io.is_colab() and self.iter % 10 == 0) or (
            io.is_colab() and self.iter % 100 == 0
        )

    def train_one_iter(self):
        # Train one iteration
        iter_time = time.time()
        losses = self.onTrainOneIter()
        iter_time = time.time() - iter_time

        self.loss_history.append([float(loss[1]) for loss in losses])

        # If you need to save the preview history
        if self.should_save_preview_history():
            plist = []

            # Processing in the Colab environment
            if io.is_colab():
                previews = self.get_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [
                        (
                            bgr,
                            self.get_strpath_storage_for_file(
                                "preview_%s.jpg" % (name)
                            ),
                        )
                    ]

            if self.write_preview_history:
                previews = self.get_history_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    path = self.preview_history_path / name
                    plist += [(bgr, str(path / (f"{self.iter:07d}.jpg")))]
                    if not io.is_colab():
                        plist += [(bgr, str(path / ("_last.jpg")))]

            if len(plist) != 0:
                self.get_preview_history_writer().post(
                    plist, self.loss_history, self.iter
                )

        self.iter += 1

        return self.iter, iter_time

    def pass_one_iter(self):
        # Perform an iteration
        self.generate_next_samples()

    def finalize(self):
        # Closing the training session
        nn.close_session()

    def is_first_run(self):
        # Determine if this is the first run
        return self.iter == 0 and not self.reset_training

    def is_debug(self):
        # Determine if you are in debug mode
        return self.debug

    def set_batch_size(self, batch_size):
        # Setting the batch size
        self.batch_size = batch_size

    def get_batch_size(self):
        # Get batch size
        return self.batch_size

    def get_iter(self):
        # Get the current iteration count
        return self.iter

    def get_author_name(self):
        # Get the current iteration count
        return self.author_name

    def set_iter(self, iter):
        # Set the number of iterations and update the loss history
        self.iter = iter
        self.loss_history = self.loss_history[:iter]

    def get_loss_history(self):
        # Get Loss History
        return self.loss_history

    def set_training_data_generators(self, generator_list):
        # Setting up the training data generator
        self.generator_list = generator_list

    def get_training_data_generators(self):
        # Get training data generator
        return self.generator_list

    def get_model_root_path(self):
        # Get model root path
        return self.saved_models_path

    def get_strpath_storage_for_file(self, filename):
        # Get the string path of the stored file
        return str(self.saved_models_path / (self.get_model_name() + "_" + filename))

    def get_strpath_configuration_path(self):
        # Get the string path of the configuration file
        return str(self.config_file_path)

    def get_strpath_def_conf_file(self):
        # Get the path to the default configuration file
        if Path(self.config_training_file).is_file():
            return str(Path(self.config_training_file))
        elif Path(
            self.config_training_file
        ).is_dir():  # For backward compatibility, if it is a directory return the directory def_conf_file.yaml
            return str(Path(self.config_training_file) / "def_conf_file.yaml")
        else:
            return None

    def get_summary_path(self):
        # Get the path to the summary file
        return self.get_strpath_storage_for_file("summary.txt")

    def get_model_conf_path(self):
        # Get the path to the model configuration file
        return self.get_strpath_storage_for_file("configuration_file.yaml")

    def get_summary_text(self, reduce_clutter=False):
        # Generate text summaries of model hyperparameters
        # visible_options are the ones that are shown, not necessarily all of them, and do not affect this training.
        visible_options = self.options.copy()
        # Parameter override, refresh the corresponding keys of the original dictionary with the new partial dictionary (which does not need to be complete), can be added or changed
        visible_options.update(self.options_show_override)

        #Hanalize the parameter values
        def str2cn(option):
            return str(option)
            # if str(option) == "False" or str(option) == "n":
            #     return "关"
            # elif str(option) == "True" or str(option) == "y":
            #     return "开"
            # else:
            #     return str(option)

        if self.model_class_name == "XSeg":
            xs_res = self.options.get("resolution", 256)
            table = PrettyTable(
                ["Resolution", "model author", "batch size", "pre-training mode"]
            )
            table.add_row(
                [
                    str(xs_res),
                    str2cn(self.author_name),
                    str2cn(self.batch_size),
                    str2cn(self.options["pretrain"]),
                ]
            )    
            # Print Forms
            summary_text = table.get_string()
            return summary_text
            
        elif self.model_class_name =="ME":
            # Create an empty table object and specify column names
            table = PrettyTable(
                ["Model Summary", "Enhancement Options", "Switch", "Parameter Settings", "Values", "Native Configuration"]
            )

            # Adding data rows
            table.add_row(
                [
                    "",
                    "Retraining high loss samples",
                    str2cn(self.options["retraining_samples"]),
                    "Batch size",
                    str2cn(self.batch_size),
                    "AdaBelief Optimizer."+str2cn(self.options["adabelief"]),
                ]
            )
            table.add_row(
                [
                    "Model name: " + self.get_model_name(),
                    "Random Flip SRC",
                    str2cn(self.options["random_src_flip"]),
                    "",
                    "",
                    "Optimizer onto the GPU: "+str2cn(self.options["models_opt_on_gpu"]),
                ]
            )
            table.add_row(
                [
                    "model author: " + self.get_author_name(),
                    "Random Flip DST",
                    str2cn(self.options["random_dst_flip"]),
                    "learning rate",
                    str2cn(self.options["lr"]),
                    "",
                ]
            )
            table.add_row(
                [
                    "",
                    "Masking Training",
                    str2cn(self.options["masked_training"]),
                    "True face (src) intensity",
                    str2cn(self.options["true_face_power"]),
                    "",
                ]
            )       
            table.add_row(
                [
                    "iteration number (math.): " + str2cn(self.get_iter()),
                    "Eyes first.",
                    str2cn(self.options["eyes_prio"]),
                    "background (src) intensity",
                    str2cn(self.options["background_power"]),
                    "Target number of iterations: " + str2cn(self.options["target_iter"]),
                ]
            )
            table.add_row(
                [
                    "model architecture: " + str2cn(self.options["archi"]),
                    "Mouth first.",
                    str2cn(self.options["mouth_prio"]),
                    "Face (dst) intensity",
                    str2cn(self.options["face_style_power"]),
                    "",
                ]
            )
            table.add_row(
                [
                    "",
                    "Side face optimization",
                    str2cn(self.options["uniform_yaw"]),
                    "Background (dst) intensity",
                    str2cn(self.options["bg_style_power"]),
                    "",
                ]
            )    
            table.add_row(
                [
                    "Resolution." + str2cn(self.options["resolution"]),
                    "Mask Edge Blur",
                    str2cn(self.options["blur_out_mask"]),
                    "Color Conversion Mode",
                    str2cn(self.options["ct_mode"]),
                    "Enable Gradient Cropping: "+str2cn(self.options["clipgrad"]),
                ]
            )
            table.add_row(
                [
                    "Autoencoder (ae_dims): " + str2cn(self.options["ae_dims"]),
                    "Decline in utilization learning rate",
                    str2cn(self.options["lr_dropout"]),
                    "",
                    "",
                    "",
                ]
            )
            table.add_row(
                [
                    "Encoder (e_dims): " + str2cn(self.options["e_dims"]),
                    "random distortion",
                    str2cn(self.options["random_warp"]),
                    "Loss function",
                    str2cn(self.options["loss_function"]),
                    "",
                ]
            )
            table.add_row(
                [
                    "Decoder (d_dims): " +  str2cn(self.options["d_dims"]),
                    "Random color (hsv micro change)",
                    str2cn(self.options["random_hsv_power"]),
                    "",
                    "",
                    "Record the history of the preview map: " + str2cn(self.options["write_preview_history"]),
                ]
            )      
            table.add_row(
                [
                    "Decoder mask (d_mask): " +  str2cn(self.options["d_mask_dims"]),
                    "Random color (no change in brightness)",
                    str2cn(self.options["random_color"]),
                    "gan_power",
                    str2cn(self.options["gan_power"]),
                    "",
                ]
            )   
            table.add_row(
                [
                    "",
                    "Random downsampling",
                    str2cn(self.options["random_downsample"]),
                    "gan_patch_size",
                    str2cn(self.options["gan_patch_size"]),
                    "",
                ]
            )   
            table.add_row(
                [
                    "Using fp16:"+str2cn(self.options["use_fp16"]),
                    "Randomly adding noise",
                    str2cn(self.options["random_noise"]),
                    "gan_dims",
                    str2cn(self.options["gan_dims"]),
                    "Automatic Backup Interval: " + str2cn(self.options["autobackup_hour"]) + " hourly",
                ]
            )      
            table.add_row(
                [
                    "",
                    "randomly generated ambiguity",
                    str2cn(self.options["random_blur"]),
                    "gan_smoothing",
                    str2cn(self.options["gan_smoothing"]),
                    "",
                ]
            )           
            table.add_row(
                [
                    "pre-training mode:" +  str2cn(self.options["pretrain"]),
                    "Randomized compressed jpeg",
                    str2cn(self.options["random_jpeg"]),
                    "gan_noise",
                    str2cn(self.options["gan_noise"]),
                    "Maximum number of backups: " + str2cn(self.options["maximum_n_backups"]),
                ]
            )    
            table.add_row(
                [
                    "",
                    "Supertwist",
                    str2cn(self.options["super_warp"]),
                    "",
                    "",
                    "",
                ]
            )    
            # Set alignment (optional) ["Model Summary", "Enhancement Options", "Switches", "Parameter Settings", "Values", "Native Configuration"]
            table.align["model summary"] = "l"  # left justification
            table.align["Enhanced Options"] = "r"  # center alignment
            table.align["switchgear"] = "l"  # center alignment
            table.align["parameterization"] = "r"  # center alignment
            table.align["numerical value"] = "l"  # center alignment
            table.align["Local Configuration"] = "r"  # center alignment
            # printable form
            summary_text = table.get_string()
        
            return summary_text
        else:
            summary_text = "undefined form"
            return summary_text

    @staticmethod
    def get_loss_history_preview(loss_history, iter, w, c, lh_height=100):
        # Converting Loss History to NumPy Arrays
        loss_history = np.array(loss_history.copy())

        # Creating Loss History Images
        lh_img = np.ones((lh_height, w, c)) * 0.1

        if len(loss_history) != 0:
            loss_count = len(loss_history[0])
            lh_len = len(loss_history)

            # Calculation of losses per column
            l_per_col = lh_len / w
            plist_max = [
                [
                    max(
                        0.0,
                        loss_history[int(col * l_per_col)][p],
                        *[
                            loss_history[i_ab][p]
                            for i_ab in range(
                                int(col * l_per_col), int((col + 1) * l_per_col)
                            )
                        ],
                    )
                    for p in range(loss_count)
                ]
                for col in range(w)
            ]

            plist_min = [
                [
                    min(
                        plist_max[col][p],
                        loss_history[int(col * l_per_col)][p],
                        *[
                            loss_history[i_ab][p]
                            for i_ab in range(
                                int(col * l_per_col), int((col + 1) * l_per_col)
                            )
                        ],
                    )
                    for p in range(loss_count)
                ]
                for col in range(w)
            ]

            # Calculate the maximum loss value for normalization
            plist_abs_max = np.mean(loss_history[len(loss_history) // 5 :]) * 2


            for col in range(0, w):
                # Iterate over each loss function
                for p in range(0, loss_count):
                    # Set the color of the data points, generated according to the HSV color space
                    point_color = [1.0] * 3
                    # loss_count=2 , p=0 or 1
                    #point_color[0:3] = colorsys.hsv_to_rgb(p * (1.0 / loss_count), 1.0, 0.8)
                    point_color_src=(0.0, 0.8, 0.9)
                    point_color_dst=(0.8, 0.3, 0.0)
                    point_color_mix=(0.1, 0.8, 0.0)
                    # According to the experiment, it should be the order of BGR
                    if p==0:
                        point_color=point_color_dst
                    if p==1:
                        point_color=point_color_src
                    # Calculate the position of the data point in the image (maximum and minimum values)
                    ph_max = int((plist_max[col][p] / plist_abs_max) * (lh_height - 1))
                    ph_max = np.clip(ph_max, 0, lh_height - 1)
                    ph_min = int((plist_min[col][p] / plist_abs_max) * (lh_height - 1))
                    ph_min = np.clip(ph_min, 0, lh_height-1)  # Limit the minimum value to the image height range

                    # Iterate over the range from the minimum to the maximum value
                    for ph in range(ph_min, ph_max+1):
                        # Adding marker points to the image array based on the calculated position and color
                        # Note: Since the origin of the array is usually in the upper left corner, you need to use (lh_height-ph-1) to convert the y coordinate to the array index
                        if p==0:
                            lh_img[(lh_height-ph-1), col] = point_color
                        if p==1:
                            current_point_color = lh_img[(lh_height-ph-1), col]
                            # Overlay new color to current color
                            #final_color = [min(1.0, current_point_color[i] + point_color_src[i]) for i in range(3)]
                            #lh_img[(lh_height-ph-1), col] = final_color
                            if (current_point_color == point_color_dst).all():
                                lh_img[(lh_height-ph-1), col] = point_color_mix
                            else:
                                lh_img[(lh_height-ph-1), col] = point_color_src
                                

        lh_lines = 8
        # Calculate the height of each row
        lh_line_height = (lh_height-1)/lh_lines
        
        # Setting Line Color and Transparency
        line_color = (0.2, 0.2, 0.2)  # gray
        
        for i in range(0,lh_lines+1):
            # Get the index of the line where the current splitter is located
            line_index = int(i * lh_line_height)
            # Set the pixel value of the current line to the line color and transparency
            lh_img[line_index, :] = line_color
            # original gray   lh_img[ int(i*lh_line_height), : ] = (0.8,)*c

        # Calculate the height position of the last line of text
        last_line_t = int((lh_lines-1)*lh_line_height)
        last_line_b = int(lh_lines*lh_line_height)

        lh_text = 'Iterations: %d iter' % (iter) if iter != 0 else ''
        
        lh_img[last_line_t:last_line_b, 0:w] += imagelib.get_text_image(
            (last_line_b - last_line_t, w, c), lh_text, color=[0.8] * c
        )
        return lh_img


class PreviewHistoryWriter:
    def __init__(self):
        # Initialization creates a multi-process queue and a handler process
        self.sq = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.process, args=(self.sq,))
        self.p.daemon = True  # Setting the process as a daemon
        self.p.start()

    def process(self, sq):
        # Handler function to process items in the queue
        while True:
            while not sq.empty():
                # Getting items from the queue
                plist, loss_history, iter = sq.get()

                # Cache Preview Loss History Images
                preview_lh_cache = {}
                for preview, filepath in plist:
                    filepath = Path(filepath)
                    i = (preview.shape[1], preview.shape[2])

                    # Get or create a loss history preview image
                    preview_lh = preview_lh_cache.get(i, None)
                    if preview_lh is None:
                        preview_lh = ModelBase.get_loss_history_preview(
                            loss_history, iter, preview.shape[1], preview.shape[2]
                        )
                        preview_lh_cache[i] = preview_lh

                    # Merge and save images
                    img = (np.concatenate([preview_lh, preview], axis=0) * 255).astype(
                        np.uint8
                    )
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2_imwrite(filepath, img)

            time.sleep(0.01)

    def post(self, plist, loss_history, iter):
        # Sending items to the queue
        self.sq.put((plist, loss_history, iter))

    # Disable serialization
    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
        self.__dict__.update(d)
