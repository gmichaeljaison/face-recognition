%%
addpath('../lib');
camlist = webcamlist;
cam = webcam(camlist{1});

%%
imgdb = imageSet('../data/mixed', 'recursive');

% featureExtractor = RandomExtractFeature();
featureExtractor = EigenFaceFeature();

[trainingFeatures, trainingLabels] = extractFeatures(imgdb, featureExtractor);

%%
faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
    'CustomBorderColor',uint8([255 255 0]));

faceRec = L1MinFaceRecognition(trainingFeatures, trainingLabels, 0.05);

figure; hold on;
set(gcf,'currentchar',' ')
while get(gcf,'currentchar')==' '
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
    
        label = predict(faceRec, faceFeature');
    
        bboxTmp = reshape(bbox, [2 2])';
        imshot = insertText(imshot, bboxTmp(1,:), label);
    
%         rectangle('Position', bbox, 'LineWidth',2, 'EdgeColor','y');
        imshot = step(shapeInserter, imshot, int16(bbox));
    end
    imshow(imshot);
    
    pause(0.01);
end