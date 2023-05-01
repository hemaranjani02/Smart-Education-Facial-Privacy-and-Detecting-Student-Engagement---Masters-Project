# FakeFaceGen
1. AttGAN-PyTorch Model
python test.py --experiment_name 256_shortcut1_inject0_none_s1 --test_int 1.0 --custom_img --custom_data --custom_attr C:\Users\Ranjani\Downloads\src_s1_FLMH\list_attr_fold1_mid_jpg.txt  --gpu
2. Fawkes Model
pip install fawkes
fawkes -d C:\Users\Ranjani\Downloads\src_s1_FLMH\fold2 --mode low --gpu gpu --batch-size=8

# ImageSpliter -> image_edit.ipynb
1. AttGAN model ouput image has combined 13 attributes including input and original image. It is splitted into individual images.

# visualizer.ipynb
1. CelebA dataset is gone through data sampling techniques.

# Dataset_Analysis CelebA -> analysis.ipynb
1. Calculates the Number of Images of One Identity (30) and total number of identities (2,343) in a celebA dataset. 
2. Generates a CelebA dataset. That means CelebA dataset provides binary attribute annotations to search for matching faces. For instance, Images of bald celebrities is +1 and Non-bald is (-1).

# FakeFaceGen folder -> AWS -> Cal_mean_similiarity.ipynb
1. For Face compare similarity between fake face(source) and raw(input) faces(reference), the similarity score is calculated using AWS face recognition face compare API.
2. Here fake faces(source) are created by passing the raw faces into a either Fawkes-low mode model, AttGAN Model, and combined Fawkes-low model & AttGAN model.
3. Raw face(reference) can be identity(same face but different location or having different hair styles, eyglasses, etc.) of fake face(source)
4. Both the source and reference images are passed into AWS face compare recognition face comapre API, and it generates a .csv files showing source file path, reference file path and many columns having unmatched(100% not similarity score) otherwise shows the similarity percentage.
5. Set the threshold as 80. if the similarity score > 80, both fake face and raw input face is matched, otherwise unmatched. 
6. And then, we sample 100 images (unique) from each folder so, number of comparison loops are as follows = 4 models x 13 attributes x 100 samples = 5,200 comparisons

# Engagement Detection Model:
Daisee Dataset Preparation: The DAiSEE dataset passes into OpenFace to generate a set of action features. The dataset consists of 7348 videos with 2.2 M frames and is split into training data of 5482 videos having 1.64 M frames (~76%) and testing data of 1866 videos having 560k frames (~24%) 2. This set of action features will be a dataset to train a new engagement model. 

# Daisee_Dataset_generator.ipynb
1. Convert video into frames.

# EngagementModel -> Train.ipynb
1. Read the .CSV files having labels Boredom, Engagement, confusion, and Frustration.
2. Building column names in dataloader using dataloader.py.
3. model.py has an Classifier Engagement Model
    The MLP has an input layer with 709 neurons, four hidden layers with 256, 64, 16, and 4 neurons, respectively, and an output layer with one neuron. Adam optimizer with a Learning rate of 0.001 and Weight decay 1e-4 = (0.0001) and Epoch from 100 to 5000, step size of 100. For the classifier method, the ReLU activation function is used for the neurons in the input and hidden layers, the Cross-Entropy Loss function is used, and the Sigmoid activation function is used for the output layer. 
4. Train and Test loader is used for loading the data and set to 'cuda'.
5. Using Sklearn metric, calculate the accuracy score of MLP_CLF_CE (MLP_Classifier_CrossEntropy)

# file_format_changer.ipynb
Converts .png to .jpg images

# Matplotlib folder -> result_parser.ipynb
# Fake Fcae comaprison Barplot graph
1. CelebA Dataset is split into 90% (182000 images) as training data to train the attribute editing model, AttGAN and 10% (20000) images is used for training and to evaluate the face fake privacy performance of the proposed method using AWS Face Recognition - Face compare API.
2. Out of 20,0000, fake face images (10,000) used as source and other 10,000 for reference in AWS face comparison API. 
3. Source-fake face images are generated using three models such as Fawkes-low mode, AttGAN - 13 attributes ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebros', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', and 'Young'], and using combined model [Fawkes-low and AttGAN]. 
4. Another 10,000(reference) identity faces similar to source fake faces are used for comparison of AWS face recognition face compare API.
5. There are 4 folders created and used 100 images each for both source(fake faces) and reference(Identity faces similar to fake faces) to compare with aws face compare API.
6. Then, the average of the 4 models is calculated and bar plot graph is shown for Fawkes-low mode only and a grouped bar plot graph for easy analysis three models of Fawkes-low mode, AttGAN, and combined Fawkes-low mode and AttGAN model.

# Engagement Detection Model classifier Barplot graph
1. Engagement is calculated using percentages and compared with:
    KNN=[0.9806, 0.9394],
    SVM=[0.9547, 0.9506],
    RF=[0.9999, 0.9475],
    XGB=[0.9983, 0.9490],
    MLP=[0.9756, 0.9210]

