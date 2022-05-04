from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog

def load_data(name):
    file_path = name + '.txt'
    if name == 'train':
        with open('train.txt', 'r') as f:
            content = f.read().split('\n') # 讀取檔案內容
            content = [i.split(' ') for i in content][:-1]
            label = []
            feature = []
            for i in tqdm(range(len(content))):
                sample = random.randint(0, len(content))
                content[sample][0]

                # One Hot Encodding
                img_label = np.zeros(50)
                img_label[int(content[sample][1])] = 1
                label.append(img_label)

                # HOG feature
                img_path = content[sample][0]
                img = imread(img_path)
                resized_img = resize(img, (128,64))
                fm = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feature.append(fm)
    
        return np.array(feature), np.array(label)
    
    else:
        file_path = name + '.txt'
        with open(file_path, 'r') as f:
            content = f.read().split('\n') # 讀取檔案內容
            content = [i.split(' ') for i in content][:-1]
            label = []
            feature = []
            for img_path, lab in content:

                img_label = np.zeros(50)
                img_label[int(lab)] = 1
                label.append(img_label)

                # HOG feature
                img = imread(img_path)
                resized_img = resize(img, (128,64))
                fm = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feature.append(fm)
        
        return np.array(feature), np.array(label)
    
train_x, train_y = load_data(train)
np.save('image_feature/train_x.npy', train_x)
np.save('image_feature/train_y.npy', train_y)

val_x, val_y = load_data(val)
np.save('image_feature/val_x.npy', val_x)
np.save('image_feature/val_y.npy', val_y) 

test_x, test_y = load_data(test)
np.save('image_feature/test_x.npy', test_x)
np.save('image_feature/test_y.npy', test_y) 