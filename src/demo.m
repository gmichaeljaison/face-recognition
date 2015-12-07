%%

addpath('../lib');

%%

imgdb = imageSet('../data/lfw-small', 'recursive');

%%

% featureExtractor = EigenFaceFeature();
featureExtractor = LaplacianFace();

[images, trainingLabels] = readImageSet(imgdb);
% [testImgs, yTest] = readImageSet(test);

featureExtractor.init(images);
trainingFeatures = featureExtractor.extract(images);

%% 

 Model = fitcknn(trainingFeatures, trainingLabels, 'NumNeighbors', 1, ...
        'Distance', 'mahalanobis', 'DistanceWeight', 'inverse');

%%

faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
    'CustomBorderColor',uint8([255 255 0]));

%%

obj = VideoReader('hilary.mp4');
video = obj.read();

%%
numframes=size(video,4);
for idx=1:numframes
    
    
    frame=video(:,:,:,idx);
    
    bboxes = step(faceDetector, frame);
    if (numel(bboxes) == 0)
        imshow(frame);
        continue;
    end
    
    for i = 1 : size(bboxes,1)
        bbox = bboxes(i,:);
        faceimg = imcrop(frame, bbox);
        faceFeature = extract(featureExtractor, faceimg);
    
        label = predict(Model, faceFeature);
    
        bboxTmp = reshape(bbox, [2 2])';
        frame = insertText(frame, bboxTmp(1,:), label{1});
    
        frame = step(shapeInserter, frame, int16(bbox));
    end
    imshow(frame);
    
    
end