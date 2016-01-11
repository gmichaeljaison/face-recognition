%%

addpath('../lib');

%%

imgdb = imageSet('../data/small', 'recursive');

% featureExtractor = EigenFaceFeature();
featureExtractor = LaplacianFace();
featureExtractor.dimension = 25;
featureExtractor.eigenF.dimension = 30;

[images, yTrain] = readImageSet(imgdb);
% [testImgs, yTest] = readImageSet(test);

featureExtractor.init(images);
xTrain = featureExtractor.extract(images);

%% 

% Model = fitcknn(xTrain, yTrain, 'NumNeighbors', 1, ...
%         'Distance', 'mahalanobis', 'DistanceWeight', 'inverse');
Model = L1MinFaceRecognition(xTrain, yTrain, 0.05);

%%

faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
    'CustomBorderColor', uint8([255 255 0]), ...
    'LineWidth', 3);

%%

video = VideoReader('../data/facetrack_360p.mp4');
images = {};
while hasFrame(video)
%     images{end+1} = readFrame(video);
% end
% 
% %%
% for n = 20 : 400
%     frame = images{n};
    frame = readFrame(video);
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
    
%         bboxTmp = reshape(bbox, [2 2])';
        
        frame = insertText(frame, [bbox(1) bbox(2) + bbox(4)], label{1});
    
        frame = step(shapeInserter, frame, int16(bbox));
    end
    imshow(frame);
    
%     imwrite(frame, fullfile('../result/', strcat(num2str(n), '.jpg')));
end
