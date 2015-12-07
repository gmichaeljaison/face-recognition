function myLoopingFcn() 
global KEY_IS_PRESSED
KEY_IS_PRESSED = 0;
gcf
set(gcf, 'KeyPressFcn', @myKeyPressFcn)

camlist = webcamlist;
cam = webcam(camlist{1});
while ~KEY_IS_PRESSED
      drawnow
      disp('looping...')
      imshot = snapshot(cam);
      imshow(imshot);
end
disp('loop ended')
function myKeyPressFcn(hObject, event)
global KEY_IS_PRESSED
KEY_IS_PRESSED  = 1;
disp('key is pressed') 