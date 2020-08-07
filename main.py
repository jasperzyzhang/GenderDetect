from load_attr import load_attr
from traininig import train,test
from plot import visualize

#customized train/test size
Total_img_num = 20000
TRAINING_SAMPLES = 8000
VALIDATION_SAMPLES = 1000
Train_proportion = 0.7
Validation_proportion = 0.15
TEST_SAMPLES = 1000
TEST_SAMPLES_MASK = 100
BATCH_SIZE = 64
NUM_EPOCHS = 100

#change the main_folder to the path to the celeba folder
main_folder = "E:/face/genderdetect/"
images_folder = main_folder + "imgs/" #save all images in imgs/ folder


def main():

    #load img info from main_folder
    img_attr = load_attr(main_folder,Total_img_num,Train_proportion,Validation_proportion)

    #Model Training
    hist,model =  train(main_folder,images_folder,img_attr,'bestmodel',TRAINING_SAMPLES,VALIDATION_SAMPLES,NUM_EPOCHS,BATCH_SIZE)

    #Model Testing

    #load model pretrained(optional)
    #model.load_weights(main_folder + 'weights6.best.inc.male.hdf5')

    CelebA = 2
    Google = 3
    acc,f1 =  test(CelebA, TEST_SAMPLES, images_folder, img_attr, model)
    acc_3,f1_3 = test(Google, TEST_SAMPLES_MASK, images_folder, img_attr, model)

    #plot loss function and accuracy

    visualize(hist, main_folder)



main()