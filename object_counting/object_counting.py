import copy
import math
import cv2
import time
from ultralytics import YOLO, solutions

# Link: Object Counting: https://docs.ultralytics.com/guides/object-counting/

# Globális változó, amely jelezni fogja, ha egy signal érkezett
#signal_received = False



car_acc = 1.75 # Autó gyorsulása m/s2
other_acc = 0.75 # Busz és Kamion gyorsulása m/s2
speed = 13.89 # 50 km/h = 13.89 m/s
car_dist = 55 # Az az út amire egy autónak szüksége van ahhoz, hogy felgyorsuljon 50 km/h-ra
other_dist = 128.7 # Az az út amire a másik 2 típusnak szüksége van ahoz, hogy felgyorsuljon 50 km/h-ra
actual_dist = 1 # Az objektum távolsága a lámpától
waitTime = 0 # Az az idő amire az utolsó felismert objektum is átérne a lámpán
reaction_time = 2 # Reakció idő
i = 1

"""def signal_handler(signum, frame):
    global signal_received
    signal_received = True
    print(f"Signal {signum} received, exiting loop.")

def setup_signal_handling():
    signal.signal(signal.CTRL_BREAK_EVENT, signal_handler)
"""
def countElements(arr):
    idx = 0
    if all(element in [1, 2, 3] for element in arr):
        return [calculateTime(element, idx + 1) for idx, element in enumerate(arr)]
    else:
        raise ValueError("A lista csak 1, 2 vagy 3 értékeket tartalmazhat.")
    #return time

def calculateTime(id, idx):
    global waitTime,car_acc,other_acc,actual_dist,reaction_time,speed,car_dist

    if idx == 1:
        plus_time = 0
    else:
        plus_time = (idx - 1) * reaction_time
    if id == 1:
        if actual_dist > car_dist: # Ha messzebb van a lámpától mint amennyi út kéne hogy felgyorsúljon 50 km/h-ra
            waitTime = round(math.sqrt((2*car_dist)/car_acc)) + plus_time + (actual_dist - car_dist) / speed
        else:
            waitTime = round(math.sqrt((2*actual_dist)/car_acc)) + plus_time
        actual_dist += 5.5 # Egy autó átlagos hossza a követési távolsággal együtt
        return waitTime
    else:
        if actual_dist > other_dist:
            waitTime = round(math.sqrt((2*other_dist)/other_acc)) + plus_time + (actual_dist - other_dist) / speed
        else:
            waitTime = round(math.sqrt((2*actual_dist)/other_acc)) + plus_time
        car_dist = other_dist # A kamion vagy busz utáni út amire a felgyorsuláshoz szükség van
        car_acc = other_acc   # A kamion vagy busz utáni gyorsulás mértéke
        if id == 2:
            actual_dist += 15 # Egy busz átlagos hossza a követési távolsággal együtt
        else:
            actual_dist += 17 # Egy kamion átlagos hossza a követési távolsággal együtt
        return waitTime

def get_video_properties(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return w, h, fps

def initialize_video_writer(output_path, codec, fps, width, height):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

def initialize_object_counter(model, line_points):
    return solutions.ObjectCounter(
        view_img=True,
        reg_pts=line_points,
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

def initialize_previous_counts(classes):
    return {cls: {'IN': 0, 'OUT': 0} for cls in classes}


def process_video(model, cap, counter, video_writer, time_limit, classes_to_count):
    """
    global signal_received

    setup_signal_handling() #Még nem pontos
    """
    start_time = time.time()
    detected_objects = []
    previous_counts = initialize_previous_counts(counter.class_wise_count.keys())

    while cap.isOpened(): # A videó feldolgozása
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
        im0 = counter.start_counting(im0, tracks)

        # Ellenőrizzük az időt és a jeleket
        if (time.time() - start_time) >= time_limit:
            break

        # Ellenőrzés és listához adás
        current_counts = counter.class_wise_count
        if current_counts.keys() == previous_counts.keys():
            # Összehasonlítja az aktuális és az előző class_wise_count értékeket
            for vehicle_type in current_counts:
                if current_counts[vehicle_type]['IN'] > previous_counts[vehicle_type]['IN']:
                    if vehicle_type == 'car':
                        detected_objects.append(1)
                    elif vehicle_type == 'truck':
                        detected_objects.append(2)
                    elif vehicle_type == 'bus':
                        detected_objects.append(3)
        previous_counts = copy.deepcopy(current_counts)

        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    return detected_objects

def start_video(time_limit):
    global i
    model = YOLO("yolov8n.pt")
    if i % 2 == 1:
        cap = cv2.VideoCapture("C:\egyetem\szakdolgozat\video samples\istockphoto-1093640824-640_adpp_is.mp4")
    else:
        cap = cv2.VideoCapture("C:\egyetem\szakdolgozat\video samples\istockphoto-984847454-640_adpp_is.mp4")
    assert cap.isOpened(), "Error reading video file"
    i = i + 1
    w, h, fps = get_video_properties(cap)

    # print(w, h, fps)
    # exit(1)

    # region_points = [(0, 850), (960, 850), (960, 950), (0, 950)] # Képen megjelenő pontok kordinátái
    region_points = [(20, 260), (750, 260), (750, 410), (20, 410)]
    classes_to_count = [2, 5, 7]# A felismerni kívánt objektumok azonosítói https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
    video_writer = initialize_video_writer("object_counting_output.avi", "mp4v", fps, w, h)
    counter = initialize_object_counter(model, region_points)

    detected_objects = process_video(model, cap, counter, video_writer, time_limit, classes_to_count)
    print("Detected objects list:", detected_objects) # A sorrendet íratom ki (1 az autó, 2 a kamion, 3 a busz)

    return detected_objects

def main():
    """
    global signal_received

    setup_signal_handling() #Még nem pontos
    """
    time_limit = 30 # Kezdeti várakozás idő
    while True:
        print("\nPiros jelzés")
        array = start_video(time_limit) # Videó feldolgozásának indítása
        time_limit = round(max(countElements(array))) # A zöld jelzés hossza
        if time_limit > 60: # Hogy ne kelljen túl hosszú időt várni ezért maximum 60 mp-ig lehet zöld
            time_limit = 60
        print("\nSárga jelzés")
        time.sleep(1)
        print("\nZöld jelzés")
        array = start_video(time_limit) # Videó feldolgozásának indítása
        time_limit = round(max(countElements(array))) # A másik oldal zöld hosszának számolása
        if time_limit > 60:
            time_limit = 60
        print("\nSárga jelzés")
        time.sleep(2)


if __name__ == "__main__":
    main()

