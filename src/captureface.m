


% 
% camlist = webcamlist;
% cam = webcam(camlist{1});
% 
% 
%  faceDetector = vision.CascadeObjectDetector;
% shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
%     'CustomBorderColor',uint8([255 255 0]));
% 
% 
% imshot = snapshot(cam);
% imshow(imshot);
% 
%     bboxes = step(faceDetector, imshot);
%     if (numel(bboxes) == 0)
%         imshow(imshot);
%         continue;
%     end
%     
%     
% i=1;  
% if(i<2)
function captureface(hObject, event)
global Model
global numperson
global camlist
global cam
global faceDetector
global labelname
numperson=numperson+1;
i=1;
x = inputdlg('oops! Please enter your name:',...
             'WHO ARE YOU?', [1 50]);
labelname{numperson}=x;
mkdir(strcat('../data/owndata/','s',num2str(numperson))); % create directory with subjects number         
         
j=1;

% Capture n faces
n=20;
faceimg=cell(1,n);

while(j<=n)
imshot = snapshot(cam);
imshow(imshot);
pause(0.5);
bboxes = step(faceDetector, imshot);
    if (numel(bboxes) == 0)
        imshow(imshot);
        continue;
    end
    
    faceimg{j} = imcrop(imshot, bboxes(1,:));
    j=j+1;
    
end
% Take 15 random faces of new subject among n captured faces
idx=randperm(n,15);         
% save the faces
for j=1:15
    imwrite(faceimg{idx(j)},strcat('../data/owndata/','s',num2str(numperson),'/',num2str(j),'.jpg'));
end
% i=i+1;

imgdb = imageSet('../data/owndata', 'recursive');
Model = learnRecognition(imgdb);
end
