%%
addpath('../lib');
camlist = webcamlist;
cam = webcam(camlist{1});

%%
imgdb = imageSet('../data/small', 'recursive');

% featureExtractor = RandomExtractFeature();
featureExtractor = EigenFaceFeature();
% featureExtractor = LaplacianFace();

[images, yTrain] = readImageSet(training);
[testImgs, yTest] = readImageSet(test);

featureExtractor.init(images);
xTrain = featureExtractor.extract(images);

%%
faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
    'CustomBorderColor',uint8([255 255 0]));

faceRec = L1MinFaceRecognition(xTrain, yTrain, 0.05);

figure; hold on;
set(gcf,'currentchar',' ')
while get(gcf,'currentchar') == ' '
% while (true)
    imshot = snapshot(cam);
    imshot = imresize(imshot, 0.3);
    
    bboxes = step(faceDetector, imshot);
    if (numel(bboxes) == 0)
        imshow(imshot);
        continue;
    end
    
    for i = 1 : size(bboxes,1)
        bbox = bboxes(i,:);
        faceimg = imcrop(imshot, bbox);
        faceFeature = extract(featureExtractor, faceimg);
    
        label = predict(faceRec, faceFeature);
    
        bboxTmp = reshape(bbox, [2 2])';
        imshot = insertText(imshot, bboxTmp(1,:), label{1});
    
        imshot = step(shapeInserter, imshot, int16(bbox));
    end
    imshow(imshot)
end

clear i bbox bboxes bboxTmp faceimg
