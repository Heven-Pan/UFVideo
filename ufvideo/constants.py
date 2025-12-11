CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 32
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"

TEMPORAL_TOKEN_FORMAT = '<TEMP-{:03d}>'

# # Time arguments
# DEFAULT_TIME_START_TOKEN = "<time_start>"
# DEFAULT_TIME_END_TOKEN = "<time_end>"

# # Box arguments
# DEFAULT_BOX_START_TOKEN = "<box_start>"
# DEFAULT_BOX_END_TOKEN = "<box_end>"

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
}

QUESTION_LIST = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]