%%
% camlist = webcamlist;
% cam = webcam(camlist{1});

%%
% imgdb = imageSet('../data/owndata', 'recursive');
% Model = learnRecognition(imgdb);

%%
% faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
    'CustomBorderColor',uint8([255 255 0]));

figure; hold on;
while (true)
    imshot = snapshot(cam);
    imshot = imresize(imshot, 0.3);
    
    bboxes = step(faceDetector, imshot);
    if (numel(bboxes) == 0)
        continue;
    end
    
    for i = 1 : size(bboxes,1)
        bbox = bboxes(i,:);
        faceimg = imcrop(imshot, bbox);
    
        faceimg = imresize(faceimg, [110 110]);
        faceFeature = extractHOGFeatures(faceimg);
    
        label = predict(Model, faceFeature);
    
        bboxTmp = reshape(bbox, [2 2])';
        imshot = insertText(imshot, bboxTmp(1,:), label);
    
%         rectangle('Position', bbox, 'LineWidth',2, 'EdgeColor','y');
        imshot = step(shapeInserter, imshot, int16(bbox));
    end
    imshow(imshot);
    
    pause(0.01);
end