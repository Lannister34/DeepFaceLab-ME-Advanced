# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Shennong

## [1.8.0] - 16/03/2024
### Added
- Comprehensive reconstruction of the display format of the summary, including tidying up the txt format.

## [1.7.0] - 13/03/2024
### Added
- Fully modified yaml one-click training parameters, possibly enabling hot modification during training.
- Intentions to create a Chinese version of yaml, abandoning bilingual coexistence in the general classification, where specific keys must be in English. 
- Acknowledgement of the bilingual display in other areas, urging users to skillfully modify yaml or rely on the developer or a friend to create a UI.

## [1.6.2] - 28/02/2024
### Changed
- Modified the path error in the training bat

## [1.6.1] - 27/02/2024
### Added
- Sinicized 2 synthesizers. One of the original synthesizers used the picture of Cat's Chineseization.
- SAEHD now supports two synthesizers.

## [1.6.0] - 26/02/2024
### Fixed
- Fixed some errors in bat content and added some text introduction.

### Improved
- Simplified and merged the functions of "Model Training -- Train Models". For example, AMP no longer distinguishes between SRC-SRC to avoid misleading newcomers, essentially just modifying the path in bat.
- Automatically reads and writes configuration files, enabling one-click training. Merged them into regular training and replaced the 2-second wait with an inquiry.

### Added
- Determined that the A-card can use ME and SAEHD training in full. This version supports an all-in-one setup for 3 graphics card versions x 2 model architectures, with RG. Modified the ME branch of the A-card training code accordingly.
- Warm tip: cc-aug color migration mode is highly recommended for its effectiveness. However, this function can't be optimized with RG, so users may need to switch to DX12, even if they are using an N card.

## [1.5.4] - 05/02/2024
### Fixed
- Fixed a bug causing the bat menu to stretch in model application.
- Corrected the environment called when using the SAEHD face synthesis bat.
- Export dfm bat now links to the corresponding environment and corrects the model architecture name.

## [1.5.2] - 01/02/2024
### Added
- ME and SAEHD model training now fully supports A-card.
  - Note: ME's A-card training still needs testing; ensuring no errors is the initial step. Eye and mouth training are currently invalid and require testing assistance.

## [1.5.1] - 31/01/2024
### Fixed
- Corrected a path error in exporting dfm for three models.
- Resolved a bat error in ME synthesis.

## [1.5.0] - 30/01/2024
### Changed
- Updated environment; Landmarks auto-error recognition should no longer report errors.

### Fixed
- Resolved a long-standing issue where renaming the model caused errors. Implemented a prompt for the error while allowing the script to continue executing instead of terminating.

### Planned
- The next version aims to simplify the optional parameters for pre-training.

- In the upcoming version, efforts will be made to explicitly indicate whether OOM (Out of Memory) is due to a lack of video or virtual memory, providing a suggested hint. This aims to improve user-friendliness, as jumping out to several pages of code can be confusing for newcomers.

## [1.4.2] - 29/01/2024
### Added
- Continued to unlock some parameter restrictions in pre-training, such as RW (random warping), while disabling some parameters.

### Fixed
- Discovered that FP16 was not completely enabled. The missing part has been addressed.

### Planned
- In the next version, consider implementing guidance based on whether it's the first run or not.

### Note
- Reminder: If you are an A-card user, you may only be able to use SAEHD training after switching environments, and you can't use ME. If you are an A-card user, please chat with me privately to exchange ideas.

## [1.4.1] - 28/01/2024
### Added
- Removed the "pre-training" process, as it's not commonly used.
- According to the official default pre-training conversion, no longer forced to clear the inter and iteration number, but ask the user.
- Opened FP16 option. Users can now choose to enable FP16. Remember to enter "?" to see the description of this parameter.
- Explained in detail the role of each parameter -udtc. Please enter "?" in the training console for details.

### Fixed
- Due to the reopening of FP16, the GAN doesn't report errors anymore. However, the ME version of GAN is said to be able to select only 1 GPU number.
- The path in the bat of Landmarks auto-error detection was wrong. It has been corrected.

## [1.4.0] - 25/01/2024
### Major Update
- N cards support RG optimization! Smaller video memory can run bigger models, or increase BS cap. FP16 is still disabled, sacrificing effects for speed. Dev team partners want to work on mixed precision.

### Added
- ME and DFL both frameworks support N-card RG while A-card doesn't, tried it! DFL before in order to everyone unified can use, is to use the DX12 version, so the N card efficiency has a discount. Now it has been stripped away, and the driver type is selected by numbers 1 and 2 during training!
- This integration package combines 12 DFL versions in one. But no significant increase in the size of the installation package. (Includes original, ME, overlay whether RG or not, then multiply by 20 series 30 series and A-card three installers). (ME_A card version may have error when RG, please contact me, after all, A card users are too few).
- Built-in Landmarks automatic error recognition, aligned merge tool, the location of the bat. Category 8: --[ Other Tests -- Extra Function ]-- Below.

### Fixed
- Due to the lack of fp16 option, it causes the error of opening gan, so I forgot to fix it. Please see the top of the comment section for the solution.

## [1.3.2] - 18/01/2024
### Added
- Corrected some bat errors, added ME model compositing and exporting.
- Already working on how to minimize the compression of the training set. The current src is saved in the recovery data is very accumulated, can utilize it in the future!

### Fixed
- Fixed an unknown error: model renaming reported an error. This was solved by rolling back the version.

### Changed
- Last 1.3 version of SAEHD was not fully native, this time it is!
- Pre-training set has been moved to workspace because it often needs to be exchanged with the aligned folder.
- Xseg model has a separate directory in workspace. I don't want to mix it with the face model.

### Updated
- Have gone along with RO optimization, eye model, AVATAR model, multi-GPU (more than 3 cards) training. Also took some time to collect, the source code have got it all.

## [1.3.0] - 11/01/2024
### Added
- Solved the original SAEHD grafting BUG.
- Added a model conversion bat, currently only from the original version to upgrade ME!
  
### Changed
- Due to the DFL to ME is easy to ascending version is not easy to descending version, for the model name to do the distinction, lest misuse. SAEHD-ME is abbreviated as SAEME model or ME model!

## [1.2.3] - 10/01/2024
### Failed Version (Records Only)
- Implanted the original DFL model, but something went wrong.
- Continuing tomorrow, main error reported is EYE_MOUTH.
- The next version may rename the ME version of SAEHD to SAEME to avoid random conversion.
- The next version will add the conversion bat between original and ME.

## [1.2.2] - 09/01/2024
### Added
- Support for A card, and comes with a bat for switching graphics cards
- Provide option when initializing: can remove Buddha statue and cat

### Changed
- Starting from version 1.2.1, it's not a patch, but a complete integration package. Note: 10-series, 20-series, 30-series N cards; DX12 are all integrated into the same package.

### Removed
- Completely remove the fp16 option

### Fixed
- Ada locks up when upgrading from the original version
- Disabled ada option after creating a model (as in, tris can't be changed)

### Updated
- Keep the native environment (based on DX12 version) and add jsonschema and attr.

## [1.1.1] - 05/01/2024
### Changed
- Changed the environment pointed to by the bat of "export loss" to Python 3.9
- Added attr dependency to the Python 3.6.8 native environment

## [1.1.0] - 04/01/2024
### Added
- Change the welcome interface plus two cats
- 5 bat files to modify the wrong pre-training path “pretrain_Celeb”.
- Loss export (it's better to let everyone know than only a few people know)
- Python368 original environment has been supplemented with jsonschema
- Added a disclaimer

### Updated
- Update tensorflow and cuda, cudnn again (old graphics card did not speed up)

## [1.0.0] - 29/12/2023
### Initialized
- Usage: Overwrite to original installation directory
- Introduction: The original authors are Cioscos, seranus, Payuyi, AnkurSaini07 of the MachineEditor organization.
    - Note: keaidelaohu did not participate in the development of Me, he just copied the source code, changed the author's name without attribution, and encrypted it. This is a violation of the GPL 3.0 open-source agreement.
    - Note: In GitHub, searching through the [Deepfacelab] keywords can yield only one in a thousand results. And the ones you guys search out are likely to be the product of non-martiality! Because you guys don't know how to search for forked projects that follow the rules, like MachineEditor. Search keywords [deepfacelab fork:true] only then you can find the good stuff!
- MVE features: new parameters (including but not limited to):
    - [n] Use fp16 ( y/n ? :help ): Don't turn this on straight away, the model may crash!
    - [n] Eyes priority ( y/n ? :help ): separate eye training
    - [n] Mouth priority ( y/n ? :help ): Separate eye training. :help ) :separate mouth training
    - [SSIM] Loss function ( SSIM/MS-SSIM/MS-SSIM L1 ? :help ): This seems to compute the loss by the face similarity algorithm, search for the term yourself, just keep the default.
    - [5e-05] Learning rate ( 0.0 ... 1.0 ? :help ): Learning rate is usually left untouched, but you can control the learning rate down manually, for example, I change it to 3e-05 when the loss reaches 0.25.
    - [n] Enable random downsample of samples ( y/n ? :help ): Same as random distortion, enhance generalization: randomly reduce resolution.
    - [n] Enable random noise added to samples ( y/n ? :help ): Same as random distortion, enhance generalization: random downsampling. :help ) :Same as random distortion, generalization enhancement: random noise map
    - [n] Enable random blur of samples ( y/n ? :help ): Enable random blur of samples ( y/n ?help ) :Enable random blur of samples ( y/n ?help ) :Same as random distortion, enhance generalization
    - [n] Enable random jpeg compression of samples ( y/n ?help ): Same as random distortion, enhance generalization: random blur map. :help ) :Same as random distort, enhance generalization: random compression of image quality.
    - [none] Enable random shadows and highlights of samples: more advanced random, enhance generalization: random simulation of light and shadow training
    - Added light and shadow color learning algorithms: “fs-aug”, “cc-aug”
- Workload:
    1. Chinese localization
    2. Upgrade python3.68 to 3.9.18
    3. Upgraded tensorflow-gpu2.6.0(or 2.4.0) to tensorflow-gpu2.10.0
    4. Dynamic bat interaction script in the main directory
- GitHub Link: [curios-city/DeepFaceLab](https://github.com/curios-city/DeepFaceLab)
    - Note: The link has been put at the top for free, the spirit stone is just a voluntary reward!

## Future Version Previews:
1. Completed
2. Completed
3. Completed
8. Completed
9. Completed
4. First do a good job of training the various stages of preset yaml one-click training, and then in the future to make a fully automated hosting
5. Increase 256 or more resolution mask training, application, synthesis
6. Extreme compression of PAK files
7. New model “Shennong frame structure V1.0”.

# MachineEditor

## [1.2.0] - 10/02/2022
### Added
-  New custom saving time; default is 25 minutes - DeepFake ENG ITA
### Updated
 - Added random blur to shadow augmentation
 - [Splittable random shadows](https://github.com/MachineEditor/DeepFaceLab/commit/9bf3eb5851c6d0fbd2cea201332a60b047bb9113)
 - Added a handful of checks to indicate that the dataset is in zip or pak form when the functions used require a bulk dataset
### Fixed
- [true face power only avaible when gan power > 0](https://github.com/MachineEditor/DeepFaceLab/commit/c833e2a2642e993409018e1f92c2565739056024)
 - [wrong def. cpu cap](https://github.com/MachineEditor/DeepFaceLab/commit/18afb868bf486e4dd4bf5eba5d41fb14a5925620)
- Other random fixes

## [1.1.0] - 29/12/2021
### Added
 -  'random_hsv_power' from Official fork - seranus
 - New 'force_full_preview' to force to do not separate dst, src and pred views in different frames 	 - randomfaker
 ### Updated
 - Refactored two pass splitting it into 3 mode: None, face, face + mask - randomfaker
 - Updated shadow augmentation splitting it in: None, src, dst, all - DeepFake ENG ITA
 - [Update requirements-colab.txt](https://github.com/MachineEditor/DeepFaceLab/commit/bfaf6255ba5c70d831151099c67b65d87a9f5466)
### Fixed
- [config-training-file supports now files](https://github.com/MachineEditor/DeepFaceLab/commit/424469845960b06652af81e77409c70a6aa73003)

## [1.0.0] - 10/12/2021
### Initialized
We created this fork from several other forks of DeepFaceLab.
Many features of this fork comes mainly from [JH's fork](https://github.com/faceshiftlabs/DeepFaceLab).
#### Features from JH's fork
- [Web UI for training preview](doc/features/webui/README.md)
- [Random color training option](doc/features/random-color/README.md)
- [Background Power training option](doc/features/background-power/README.md)
- [MS-SSIM loss training option](doc/features/ms-ssim)
- [GAN label smoothing and label noise options](doc/features/gan-options)
- MS-SSIM+L1 loss function, based on ["Loss Functions for Image Restoration with Neural Networks"](https://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks)
- Autobackup options:
	- Session name
	- ISO Timestamps (instead of numbered)
	- Max number of backups to keep (use "0" for unlimited)
- New sample degradation options (only affects input, similar to random warp):
	- Random noise (gaussian/laplace/poisson)
	- Random blur (gaussian/motion)
	- Random jpeg compression
	- Random downsampling
- New "warped" preview(s): Shows the input samples with any/all distortions.
#### Features from other forks
- FaceSwap-Aug in the color transfer modes
- Custom face types
#### Features from MVE Development team
- External configuration files by [Cioscos](https://github.com/Cioscos) aka DeepFake ENG ITA
	- use --auto_gen_config CLI param to auto generate config. file or resume its configuration
	- use --config_training_file CLI param external configuration file override
- Tensorboard support by [JanFschr](https://github.com/JanFschr) aka randomfaker
- AMP training updates - DeepFake ENG ITA & randomfaker
- shadow augmentation (needs testing to see if it can generalise well) - randomfaker
- filename labels by [Ognjen](https://github.com/seranus) aka JesterX aka seranus
- zip faceset support - randomfaker
- exposed new configuration parameters (cpu, lr, preview samples)
- Added pre-sharpen into the merger. It helps the model to fit better to the target face. Idea taken from [DeepFaceLive](https://github.com/iperov/DeepFaceLive)
- Added two pass option into the merger. It processes the generated face twice. Idea taken from [DeepFaceLive](https://github.com/iperov/DeepFaceLive)

[1.2.0]: https://github.com/MachineEditor/DeepFaceLab/tree/a7e0cbb0295ae35e9098ab383bc6e0a8bdd0f944
[1.1.0]: https://github.com/MachineEditor/DeepFaceLab/tree/bfaf6255ba5c70d831151099c67b65d87a9f5466
[1.0.0]: https://github.com/MachineEditor/DeepFaceLab/tree/6c5a5934452e174779561885fccf3f1ed38be9ae
