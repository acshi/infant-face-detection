// InfantFaceDetection.cpp : Defines the entry point for the console application.
//

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/data_io/load_image_dataset.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/shape_predictor.h>
#include <iostream>

using namespace dlib;
using namespace std;

double interocular_distance(const full_object_detection& det) {
    dlib::vector<double, 2> l, r;
    double cnt = 0;
    // Find the center of the eye on the left by averaging its points
    for (unsigned long i = 0; i <= 2; ++i) {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    // Find the center of the eye on the right by averaging its points
    cnt = 0;
    for (unsigned long i = 3; i <= 5; ++i) {
        r += det.part(i);
        ++cnt;
    }
    r /= cnt;

    // Now return the distance between the centers of the eyes
    return length(l - r);
}

std::vector<std::vector<double> > get_interocular_distances(const std::vector<std::vector<full_object_detection>>& objects) {
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i) {
        for (unsigned long j = 0; j < objects[i].size(); ++j) {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

std::vector<image_window::overlay_circle> render_eye_detections(const full_object_detection& det) {
    const rgb_pixel color = rgb_pixel(0, 255, 0);
    std::vector<image_window::overlay_circle> circles;
    for (unsigned long i = 0; i < 6; ++i) {
        circles.push_back(image_window::overlay_circle(det.part(i), 3, color));
    }
    return circles;
}

void exhibit_face_detections(dlib::array<array2d<unsigned char>>& imgs, std::vector<std::vector<full_object_detection>>& allFaces) {
    image_window winTest;

    for (uint32_t i = 0; i < imgs.size(); i++) {
        std::vector<full_object_detection> faces = allFaces[i];
        for (uint32_t j = 0; j < faces.size(); j++) {
            full_object_detection face = faces[j];

            winTest.clear_overlay();
            winTest.set_image(imgs[i]);
            winTest.add_overlay(render_eye_detections(face));
            winTest.add_overlay(image_window::overlay_rect(faces[j].get_rect(), rgb_pixel(0, 255, 0)));
            cin.get();
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "usage: " << argv[0] << " [path to image and labels xml file]" << endl;
        return 0;
    }
    
    dlib::array<array2d<unsigned char>> allImages;
    std::vector<std::vector<full_object_detection>> allFullFaces;
    load_image_dataset(allImages, allFullFaces, argv[1]);
    
    dlib::array<array2d<unsigned char>> trainingImages;
    dlib::array<array2d<unsigned char>> testingImages;
    
    std::vector<std::vector<rectangle>> trainingFaces;
    std::vector<std::vector<rectangle>> testingFaces;
    
    dlib::array<array2d<unsigned char>> trainingBaseImages;
    dlib::array<array2d<unsigned char>> testingBaseImages;
    
    std::vector<std::vector<full_object_detection>> trainingFullFaces;
    std::vector<std::vector<full_object_detection>> testingFullFaces;
    
    // Divide the images into two halves.
    // Make sure the left-right flip versions are either both in training or both in testing
    int imageCount = allImages.size();
    int baseImageCount = imageCount / 2;
    for (int i = 0; i < imageCount; i++) {
        std::vector<rectangle> faceRects;
        for (int j = 0; j < allFullFaces[i].size(); j++) {
            faceRects.push_back(allFullFaces[i][j].get_rect());
        }
        
        array2d<unsigned char> imgCopy;
        assign_image(imgCopy, allImages[i]);
        
        if ((i % baseImageCount) <= baseImageCount / 2) {
            trainingImages.push_back(allImages[i]);
            trainingBaseImages.push_back(imgCopy);
            trainingFullFaces.push_back(allFullFaces[i]);
            trainingFaces.push_back(faceRects);
        } else {
            testingImages.push_back(allImages[i]);
            testingBaseImages.push_back(imgCopy);
            testingFullFaces.push_back(allFullFaces[i]);
            testingFaces.push_back(faceRects);
        }
    }
    
    // take advantage of left right symmetry
    add_image_left_right_flips(trainingImages, trainingFaces);
    add_image_left_right_flips(testingImages, testingFaces);
    
    //upsample_image_dataset<pyramid_down<2> >(trainingImages, trainingFaces);
    //upsample_image_dataset<pyramid_down<2> >(testingImages,  testingFaces);
    
    cout << "num training images: " << trainingImages.size() << endl;
    cout << "num testing images:  " << testingImages.size() << endl;
    
    
    // Finally we get to the training code.  dlib contains a number of
    // object detectors.  This typedef tells it that you want to use the one
    // based on Felzenszwalb's version of the Histogram of Oriented
    // Gradients (commonly called HOG) detector.  The 6 means that you want
    // it to use an image pyramid that downsamples the image at a ratio of
    // 5/6.  Recall that HOG detectors work by creating an image pyramid and
    // then running the detector over each pyramid level in a sliding window
    // fashion.   
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
    image_scanner_type scanner;
    // The sliding window detector will be 80 pixels wide and 80 pixels tall.
    scanner.set_detection_window_size(80, 80);
    // Finally, you can add a nuclear norm regularizer to the SVM trainer.  Doing has
    // two benefits.  First, it can cause the learned HOG detector to be composed of
    // separable filters and therefore makes it execute faster when detecting objects.
    // It can also help with generalization since it tends to make the learned HOG
    // filters smoother.
    // The argument determines how important it is to have a small nuclear norm.  A
    // bigger regularization strength means it is more important.  The smaller the
    // nuclear norm the smoother and faster the learned HOG filters will be, but if the
    // regularization strength value is too large then the SVM will not fit the data
    // well.  This is analogous to giving a C value that is too small.
    //scanner.set_nuclear_norm_regularization_strength(0.001);
    
    structural_object_detection_trainer<image_scanner_type> faceTrainer(scanner);
    faceTrainer.set_num_threads(std::thread::hardware_concurrency());
    
    // The trainer is a kind of support vector machine and therefore has the usual SVM
    // C parameter.  In general, a bigger C encourages it to fit the training data
    // better but might lead to overfitting.  You must find the best C value
    // empirically by checking how well the trained detector works on a test set of
    // images you haven't trained on.  Don't just leave the value set at 1.  Try a few
    // different C values and see what works best for your data.
    faceTrainer.set_c(0.5);
    // We can tell the trainer to print it's progress to the console if we want.  
    faceTrainer.be_verbose();
    // The trainer will run until the "risk gap" is less than 0.01.  Smaller values
    // make the trainer solve the SVM optimization problem more accurately but will
    // take longer to train.  For most problems a value in the range of 0.1 to 0.01 is
    // plenty accurate.  Also, when in verbose mode the risk gap is printed on each
    // iteration so you can see how close it is to finishing the training.
    faceTrainer.set_epsilon(0.04);
    
    object_detector<image_scanner_type> faceDetector = faceTrainer.train(trainingImages, trainingFaces);
    
    // Now that we have a face detector we can test it.  The first statement tests it
    // on the training data.  It will print the precision, recall, and then average precision.
    cout << "face training results (precision, recall, average precision): " << test_object_detection_function(faceDetector, trainingImages, trainingFaces) << endl;
    // However, to get an idea if it really worked without overfitting we need to run
    // it on images it wasn't trained on.  The next line does this.  Happily, we see
    // that the object detector works perfectly on the testing images.
    cout << "face testing results (precision, recall, average precision): " << test_object_detection_function(faceDetector, testingImages, testingFaces) << endl;
    
    
    shape_predictor_trainer eyeTrainer;
    // because we have a small dataset
    eyeTrainer.set_oversampling_amount(200);
    // Reduce the capacity of the model by explicitly increasing
    // the regularization (making nu smaller) and by using trees with
    // smaller depths. 
    eyeTrainer.set_nu(0.05); // 0.05
    eyeTrainer.set_tree_depth(3);
    eyeTrainer.set_num_threads(4);
    eyeTrainer.be_verbose();

    shape_predictor eyePredictor = eyeTrainer.train(trainingBaseImages, trainingFullFaces);


    // Now that we have a model we can test it.  This function measures the
    // average distance between a face landmark output by the
    // shape_predictor and where it should be according to the truth data.
    // Note that there is an optional 4th argument that lets us rescale the
    // distances.  Here we are causing the output to scale each face's
    // distances by the interocular distance, as is customary when
    // evaluating face landmarking systems.
    cout << "mean eye training error: " <<
        test_shape_predictor(eyePredictor, trainingBaseImages, trainingFullFaces, get_interocular_distances(trainingFullFaces)) << endl;

    // The real test is to see how well it does on data it wasn't trained
    // on.  We trained it on a very small dataset so the accuracy is not
    // extremely high, but it's still doing quite good.  Moreover, if you
    // train it on one of the large face landmarking datasets you will
    // obtain state-of-the-art results, as shown in the Kazemi paper.
    cout << "mean eye testing error:  " <<
        test_shape_predictor(eyePredictor, testingBaseImages, testingFullFaces, get_interocular_distances(testingFullFaces)) << endl;

    // Finally, we save the model to disk so we can use it later.
    serialize("eyePredictor.dat") << eyePredictor;
    
    image_window win;

    for (uint32_t i = 0; i < testingBaseImages.size(); i++) {
        std::vector<rectangle> faces = faceDetector(testingBaseImages[i]);
        //std::vector<full_object_detection> faces = testingFullFaces[i];
        for (uint32_t j = 0; j < faces.size(); j++) {
            rectangle faceRect = faces[j];
            //rectangle faceRect = faces[j].get_rect();
            full_object_detection face = eyePredictor(testingBaseImages[i], faceRect);
            //for (int k = 0; k < 6; k++) {
            //    cout << face.part(k) << endl;
            //}

            win.clear_overlay();
            win.set_image(testingBaseImages[i]);
            win.add_overlay(render_eye_detections(face));
            win.add_overlay(image_window::overlay_rect(faceRect, rgb_pixel(0, 255, 0)));
            cin.get();
        }
    }
    
    return 0;
}
