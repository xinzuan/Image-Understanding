// Create the virtual camera
ImageSize = [ 2560 1280 ];
skew = 2560*tan(30*pi/180);
// Calibration matrix
K = [ 2560 skew 320;
       0 2560 240;
       0 0 1];
// Rotation matrix
thx = 0*pi/90; thy = -45*pi/90; thz = -45*pi/90; %radian
Rx = [ 1 0 0;
   0 cos(thx) -sin(thx);
   0 sin(thx) cos(thx)];
Ry = [ cos(thy) 0 sin(thy);
   0 1 0;
   -sin(thy) 0  cos(thy)];
Rz = [ cos(thz) -sin(thz) 0;
   sin(thz) cos(thz) 0;
   0 0 1];
R = Rz*Ry*Rx;
// Camera location
C_ = [-4; -1; 1]; 
// Translation vector
t = -R*C_; 
// Camera projection matrix
P = K*R*[eye(3) -C_]; //  P = K*[R t];
// Build virtual environment
// Object 1
t1 = 0:pi/500:pi;
X1 = sin(t1).*cos(10*t1)+4;
Y1 = sin(t1).*sin(10*t1)+3;
Z1 = cos(t1)+2;
Xt1 = sin(t1).*cos(12*t1)+4;
Yt1 = sin(t1).*sin(12*t1)+3;
Zt1 = cos(t1)+2;
//  Object 2
r = 0:pi/100:pi;
X2 = sin(r).*cos(10*r)+10;
Y2 = sin(r).*sin(10*r);
Z2 = cos(r);
//  Object 3
Z3 = linspace(0,4,200);
X3 = exp(-Z3./50).*sin(50*Z3)+3;
Y3 = exp(-Z3./50).*cos(50*Z3)-2;
//  Draw 3D points and world coordinate frame
figure; plot3(X1,Y1,Z1,Xt1,Yt1,Zt1,'c.');hold on;
plot3(X2,Y2,Z2, 'm.');
plot3(X3,Y3,Z3);
hold on;
grid on;
axis equal;
len = 1.5;
Xp = [len; 0; 0;]; Yp = [0; len; 0]; Zp = [0; 0; len];
plot3([0 Xp(1)], [0 Xp(2)], [0 Xp(3)], 'b-', 'LineWidth',2);
plot3([0 Yp(1)], [0 Yp(2)], [0 Yp(3)], 'g-', 'LineWidth',2);
plot3([0 Zp(1)], [0 Zp(2)], [0 Zp(3)], 'r-', 'LineWidth',2);
xlabel('X (meter)');  ylabel('Y (meter)');
zlabel('Z (meter)');  axis equal; grid on;
//  Camera coordinate frame
CX = R'*(Xp-t); 
CY = R'*(Yp-t);
CZ = R'*(Zp-t);
plot3([C_(1) CX(1)], [C_(2) CX(2)], [C_(3) CX(3)], 'b:', 'LineWidth',2);
plot3([C_(1) CY(1)], [C_(2) CY(2)], [C_(3) CY(3)], 'g:', 'LineWidth',2);
plot3([C_(1) CZ(1)], [C_(2) CZ(2)], [C_(3) CZ(3)], 'r:', 'LineWidth',2);
hold off;

//  Acquire images of 3D points
d = P*[X1;Y1;Z1; ones(1,length(X1))];
d1 = P*[Xt1;Yt1;Zt1; ones(1,length(Xt1))];
d(1,:) = d(1,:)./d(3,:);
d(2,:) = d(2,:)./d(3,:);
d1(1,:) = d1(1,:)./d1(3,:);
d1(2,:) = d1(2,:)./d1(3,:);
e = P*[X2;Y2;Z2; ones(1,length(X2))];
e(1,:) = e(1,:)./e(3,:);
e(2,:) = e(2,:)./e(3,:);
temp = [X3;Y3;Z3; ones(1,length(X3))];
f = P*temp;
f(1,:) = f(1,:)./f(3,:);
f(2,:) = f(2,:)./f(3,:);


//  Draw images of 3D points
figure;  axis equal; grid on; hold on;
plot(d(1,:),d(2,:),d1(1,:),d1(2,:),'c.');
plot(e(1,:),e(2,:),'m.');
plot(f(1,:),f(2,:));
axis([6 ImageSize(1) 6 ImageSize(2)]);
xlabel('x (pixel)');  ylabel('y (pixel)'); hold off;