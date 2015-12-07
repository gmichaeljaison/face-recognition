


function trykey()
global keypress
keypress=0;
gcf
set(gcf, 'KeyPressFcn', @interupt)
%figure();
while ~keypress
%figure();
a=1;
drawnow
disp(a);
end


end
