from PIL import Image
import os, csv
os.makedirs('data/raw', exist_ok=True)
# create 8 tiny images
for i in range(8):
    img = Image.new('L', (224,224), color=255 if i%2==0 else 0)
    img.save(f'data/raw/img_{i:03d}.png')
# create CSV
with open('data/metadata_small.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['image_path','label'])
    for i in range(8):
        w.writerow([f'raw/img_{i:03d}.png', i%2])
print('Created sample CSV and images')
