import scenedetect as sd


def detect_scenes(video_path, threshold=15.0):
    video = sd.open_video(video_path)
    sm = sd.SceneManager()
        
    sm.add_detector(sd.ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    print(f"Scene detection with threshold {threshold}: Found {len(scenes)} scenes")
    return scenes

def get_scene_frame_samples(video_path, no_of_samples, threshold=15.0):
    scenes = detect_scenes(video_path, threshold)
    scenes_frame_samples = []    
    for scene_idx in range(len(scenes)):
        scene_length = abs(scenes[scene_idx][0].frame_num - scenes[scene_idx][1].frame_num)
        every_n = round(scene_length/no_of_samples)
        local_samples = [(every_n * n) + scenes[scene_idx][0].frame_num for n in range(3)]
            
        scenes_frame_samples.append(local_samples)
    return scenes_frame_samples

if __name__ == "__main__":
    video_path = '/mnt/c/Users/Administrator/Desktop/memryx-testing/videos/sample.mp4' # path to video on machine
    no_of_samples = 3 # number of samples per scene
    scene_frame_samples = get_scene_frame_samples(video_path, 3)
    print(scene_frame_samples)

