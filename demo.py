import matplotlib
import face_alignment
matplotlib.use('Agg')
import yaml
from tqdm import tqdm
import cv2
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import gc


def load_checkpoints():
    with open("config/vox-256.yaml") as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)

    checkpoint = torch.load("vox-cpk.pth.tar", map_location=torch.device(device))

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if device == "cuda":
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    gc.collect()
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True):

    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def find_best_frame(source, driving):

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(1, flip_input=True, device="cpu")
    kp_source = fa.get_landmarks_from_image(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0

    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks_from_image(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i

    gc.collect()
    return frame_num


if __name__ == "__main__":

    driving_video = "317.mp4"
    source_image = "b.jpg"
    adapt_scale = True
    relative = False
    find_best_frame = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu = True if device == "cuda" else False

    source_image = cv2.imread(source_image)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    reader = imageio.get_reader(driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []

    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError as e:
        reader.close()
        print(e)

    reader.close()

    input_image = resize(source_image, (256, 256), anti_aliasing=True)[..., :3]  # TODO
    driving_video = [resize(frame, (256, 256), anti_aliasing=True)[..., :3] for frame in driving_video]

    generator, kp_detector = load_checkpoints()

    if find_best_frame:
        i = find_best_frame(input_image, driving_video)
        print("Best frame: " + str(i))

        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i + 1)][::-1]
        predictions_forward = make_animation(input_image, driving_forward, generator, kp_detector,
                                             relative=relative, adapt_movement_scale=adapt_scale)
        predictions_backward = make_animation(input_image, driving_backward, generator, kp_detector,
                                              relative=relative, adapt_movement_scale=adapt_scale)

        predictions = predictions_backward[::-1] + predictions_forward[1:]

    else:
        predictions = make_animation(input_image, driving_video, generator, kp_detector, relative=relative,
                                     adapt_movement_scale=adapt_scale)

    imageio.mimsave(f"{driving_video[:-4]}_adapt_{adapt_scale}_relative_{relative}.mp4", [img_as_ubyte(frame) for frame in predictions], fps=fps)
