import os
import multiprocessing
import json
input_dir = "./"
output_dir = "./new/"

os.makedirs(output_dir, exist_ok=True)
namels = [f for f in os.listdir(input_dir) if f.endswith(".json")]
def process_file(name):
    input_path = os.path.join(input_dir, name)
    output_path = os.path.join(output_dir, name)
    with open(input_path, 'r', encoding='utf-8') as f:
        data=json.load(f)
        #data = f.read().replace("battery1", "battery").replace("battery5", "battery")
    for ra in data['shapes']:
        if ra['label']=='battery1':ra['label']='battery'
        elif ra['label']=='battery5':ra['label']='battery'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data,f,indent=4)
        #f.write(data)

    print(name)

# 使用多进程加速
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=os.cpu_count())  # 使用最大 CPU 核心
    pool.map(process_file, namels)
    pool.close()
    pool.join()
