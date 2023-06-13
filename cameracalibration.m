// 1) Generate three synthetic images of a planar calibration object using three virtual cameras.
//    Three cameras must have the same intrinsic parameters but different extrinsic parameters.
// 2) Add Gaussian noise to the 2D image points.
// 3) Estimate the intrinsic and extrinsic parameters using the DLT of Zhang's method.
// 4) Refine the intrinsic and extrinsic parameters by minimizing the geometric error using LM.
// 5) Compare the geometric error before and after the LM-based refinement.



//  image size (pixel)
width = 1024; height = 768;
//  Calibration matrix
K = [200-3 0.02 width/2-5; 0 200+2 height/2+10; 0 0 1]; //  pixel
//  Rotation matrix
thx = 10*pi/180; thy = -180*pi/180; thz = -70*pi/180; //  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R1 = Rz*Ry*Rx;
thx = -10*pi/180; thy = -170*pi/180; thz = 0*pi/180; //  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R2 = Rz*Ry*Rx;
thx = 30*pi/180; thy = -190*pi/180; thz = 70*pi/180; //  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R3 = Rz*Ry*Rx;
//  Camera center in world coordinate frame (meter)
C1_ = [-2; -2; 12]; C2_ = [5; 5; 15]; C3_ = [-3; 6; 17];
//  Translation vector
t1 = -R1*C1_; t2 = -R2*C2_; t3 = -R3*C3_;
//  Camera projection matrix (P = K*[R t])
P1 = K*R1*[eye(3) -C1_];
P2 = K*R2*[eye(3) -C2_];
P3 = K*R3*[eye(3) -C3_];
//  3D points in world coordinate frame
x = -10:2:10; y = -10:2:10; z = 0;
[X Y Z] = meshgrid(x,y,z);
X = X(:); Y = Y(:); Z = Z(:);
//  Draw 3D points and world coordinate frame
figure; plot3(X,Y,Z,'k.'); hold on;
xlabel('X (meter)'); ylabel('Y (meter)'); zlabel('Z (meter)');
axis equal; grid on;
//  Draw camera coordinate frame
len = 2;
Xp = [len; 0; 0]; Yp = [0; len; 0]; Zp = [0; 0; len];
CX1 = R1'*(Xp - t1); CY1 = R1'*(Yp - t1); CZ1 = R1'*(Zp - t1);

CX2 = R2'*(Xp - t2); CY2 = R2'*(Yp - t2); CZ2 = R2'*(Zp - t2);
CX3 = R3'*(Xp - t3); CY3 = R3'*(Yp - t3); CZ3 = R3'*(Zp - t3);
plot3([C1_(1) CX1(1)],[C1_(2) CX1(2)],[C1_(3) CX1(3)],'b:','LineWidth',2);
plot3([C1_(1) CY1(1)],[C1_(2) CY1(2)],[C1_(3) CY1(3)],'g:','LineWidth',2);
plot3([C1_(1) CZ1(1)],[C1_(2) CZ1(2)],[C1_(3) CZ1(3)],'r:','LineWidth',2);
plot3([C2_(1) CX2(1)],[C2_(2) CX2(2)],[C2_(3) CX2(3)],'b-','LineWidth',2);
plot3([C2_(1) CY2(1)],[C2_(2) CY2(2)],[C2_(3) CY2(3)],'g-','LineWidth',2);
plot3([C2_(1) CZ2(1)],[C2_(2) CZ2(2)],[C2_(3) CZ2(3)],'r-','LineWidth',2);
plot3([C3_(1) CX3(1)],[C3_(2) CX3(2)],[C3_(3) CX3(3)],'b-','LineWidth',2);
plot3([C3_(1) CY3(1)],[C3_(2) CY3(2)],[C3_(3) CY3(3)],'g-','LineWidth',2);
plot3([C3_(1) CZ3(1)],[C3_(2) CZ3(2)],[C3_(3) CZ3(3)],'r-','LineWidth',2);
hold off;
//  Acquire images of 3D points
x1 = P1*[X'; Y'; Z'; ones(1,length(X))];
x1(1,:) = x1(1,:)./x1(3,:); x1(2,:) = x1(2,:)./x1(3,:);
x1(3,:) = x1(3,:)./x1(3,:);
x2 = P2*[X'; Y'; Z'; ones(1,length(X))];
x2(1,:) = x2(1,:)./x2(3,:); x2(2,:) = x2(2,:)./x2(3,:);
x2(3,:) = x2(3,:)./x2(3,:);
x3 = P3*[X'; Y'; Z'; ones(1,length(X))];
x3(1,:) = x3(1,:)./x3(3,:); x3(2,:) = x3(2,:)./x3(3,:);
x3(3,:) = x3(3,:)./x3(3,:);
//  Draw images of 3D points
figure;
subplot(131); plot(x1(1,:),x1(2,:),'r.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(132); plot(x2(1,:),x2(2,:),'b.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(133); plot(x3(1,:),x3(2,:),'k.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
save('data_calibration.mat','K','R1','t1','R2','t2','R3','t3','X','Y','Z','x1','x2','x3','width','height');

//  Load data
load('data_calibration.mat');
//  add noise
scale = 100;
a1 = x1(1:2,:) + randn(size(x1(1:2,:)))*scale; x1(3,:) = 1;
a2 = x2(1:2,:) + randn(size(x2(1:2,:)))*scale; x2(3,:) = 1;
a3 = x3(1:2,:) + randn(size(x3(1:2,:)))*scale; x3(3,:) = 1;
//  Draw images of 3D points
figure;
subplot(131); plot(x1(1,:),x1(2,:),'b.'); axis equal; grid on; hold on;
subplot(131); plot(a1(1,:),a1(2,:),'r.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(132); plot(x2(1,:),x2(2,:),'r.'); axis equal; grid on; hold on;
subplot(132); plot(a2(1,:),a2(2,:),'b.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(133); plot(x3(1,:),x3(2,:),'r.'); axis equal; grid on; hold on;
subplot(133); plot(a3(1,:),a3(2,:),'k.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
x1 = x1(1:2,:) + randn(size(x1(1:2,:)))*scale; x1(3,:) = 1;
x2 = x2(1:2,:) + randn(size(x2(1:2,:)))*scale; x2(3,:) = 1;
x3 = x3(1:2,:) + randn(size(x3(1:2,:)))*scale; x3(3,:) = 1;
//  Homography estimation for each image
XY = [X'; Y'; ones(size(X'))];
Hz = cell(1,3);
Hz{1} = estimate_homography(XY,x1,width,height);
Hz{2} = estimate_homography(XY,x2,width,height);
Hz{3} = estimate_homography(XY,x3,width,height);
//  Intrinsic parameters estimation
VM = [];
for i=1:3
   h = Hz{i};
   v11 = [h(1,1)*h(1,1), h(1,1)*h(2,1)+h(2,1)*h(1,1), h(2,1)*h(2,1), ...
   h(3,1)*h(1,1)+h(1,1)*h(3,1), h(3,1)*h(2,1)+h(2,1)*h(3,1) h(3,1)*h(3,1)];
   v22 = [h(1,2)*h(1,2), h(1,2)*h(2,2)+h(2,2)*h(1,2), h(2,2)*h(2,2), ...
   h(3,2)*h(1,2)+h(1,2)*h(3,2), h(3,2)*h(2,2)+h(2,2)*h(3,2) h(3,2)*h(3,2)];
   v12 = [h(1,1)*h(1,2), h(1,1)*h(2,2)+h(2,1)*h(1,2), h(2,1)*h(2,2), ...
   h(3,1)*h(1,2)+h(1,1)*h(3,2), h(3,1)*h(2,2)+h(2,1)*h(3,2) h(3,1)*h(3,2)];
   VM = [VM; v12; v11-v22];
end
[U D V] = svd(VM);
b = V(:,end);
v0 = (b(2)*b(4)-b(1)*b(5)) / (b(1)*b(3)-b(2)^2);
lamda = b(6)-(b(4)^2+v0*(b(2)*b(4)-b(1)*b(5)))/b(1);
alpha = sqrt(lamda/b(1));
beta = sqrt(lamda*b(1)/(b(1)*b(3)-b(2)^2));
c = -b(2)*alpha^2*beta/lamda;
u0 = c*v0/alpha - b(4)*alpha^2/lamda;
K_ = [alpha c u0; 0 beta v0; 0 0 1]; //  estimated intrinsic parameter
err_dlt_1 = calculate_error(K_,XY,x1);
err_dlt_2 = calculate_error(K_,XY,x2);
err_dlt_3 = calculate_error(K_,XY,x3);
//  Extrinsic parameters estimation
iK_ = inv(K_);
for i=1:3
   h = H{i};
   lamda = (1/norm(iK_*h(:,1)) + 1/norm(iK_*h(:,2))) / 2;
   r1 = lamda*iK_*h(:,1);
   r2 = lamda*iK_*h(:,2);
   r3 = cross(r1,r2);
   Rtemp = [r1 r2 r3];
   [U D V] = svd(Rtemp);
   R_{i} = U*V';
   t_{i} = lamda*iK_*h(:,3);
   C = -R_{i}'*t_{i};
   if C(3) < 0
       R_{i}(:,1) = -R_{i}(:,1);
       R_{i}(:,2) = -R_{i}(:,2);
       t_{i} = -t_{i};
   end
end








function H = estimate_homography(x1n,x2n,W1,H1)
   //  Normalize data
   iT1 = [W1+H1 0 W1/2; 0 W1+H1 H1/2; 0 0 1]; T1 = inv(iT1);
   iT2 = [W1+H1 0 W1/2; 0 W1+H1 H1/2; 0 0 1]; T2 = inv(iT2);
   x1nm = T1*x1n;
   x2nm = T2*x2n;
   //  Estimate homography by DLT
   A = []; numPnt = size(x1nm,2);
   for i=1:numPnt
       x = x1nm(1,i); y = x1nm(2,i); w = x1nm(3,i);
       x_ = x2nm(1,i); y_ = x2nm(2,i); w_ = x2nm(3,i);
       tmp = [0 0 0 -x*w_ -y*w_ -w*w_ x*y_ y*y_ w*y_;
       x*w_ y*w_ w*w_ 0 0 0 -x*x_ -y*x_ -w*x_];
       A = [A; tmp];
   end
   [U D V] = svd(A);
   h = V(:,end);
   Hnm = [h(1:3)'; h(4:6)'; h(7:9)'];
   %denormalize
   H = inv(T2)*Hnm*T1;
end


function refined_H = refined_using_LM(P,temp,x1)
//  refining the homography using LM

options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt', 'MaxIterations',1000,'Display', 'iter', 'MaxFunEvals', 100000, 'TolX', 1e-10, 'TolFun', 1e-40);
   //  Nonlinear least-squares solver
   stop = 1;
   [refined_h, resnorm, res] = lsqnonlin(@fun,P,[],[],options,temp,x1);
   refined_H = refined_h;

end
function F = fun(P1,temp,x1)
 
 
  
  
   x1_ = P1*temp;

   x1_(1,:) = x1_(1,:)./x1_(3,:); x1_(2,:) = x1_(2,:)./x1_(3,:);
   x1_(3,:) = x1_(3,:)./x1_(3,:);
  

   //  Calculate error (vector)
   F = [sqrt((x1_(1,:)-x1(1,:)).^2+(x1_(2,:)-x1(2,:)).^2) ];
end




//  Load data
load('data_calibration.mat');
//  Add noise
scale = 50;
rng(1,'philox')
a1 = x1(1:2,:) + randn(size(x1(1:2,:)))*scale; x1(3,:) = 1;
a2 = x2(1:2,:) + randn(size(x2(1:2,:)))*scale; x2(3,:) = 1;
a3 = x3(1:2,:) + randn(size(x3(1:2,:)))*scale; x3(3,:) = 1;
//  Draw images of 3D points
figure;
subplot(131); plot(x1(1,:),x1(2,:),'b.'); axis equal; grid on; hold on;
subplot(131); plot(a1(1,:),a1(2,:),'r.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(132); plot(x2(1,:),x2(2,:),'r.'); axis equal; grid on; hold on;
subplot(132); plot(a2(1,:),a2(2,:),'b.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
subplot(133); plot(x3(1,:),x3(2,:),'r.'); axis equal; grid on; hold on;
subplot(133); plot(a3(1,:),a3(2,:),'k.'); axis equal; grid on; hold on;
axis([1 width 1 height]); xlabel('x (pixel)'); ylabel('y (pixel)');
hold off;
rng(1,'philox')
x1 = x1(1:2,:) + randn(size(x1(1:2,:)))*scale; x1(3,:) = 1;
x2 = x2(1:2,:) + randn(size(x2(1:2,:)))*scale; x2(3,:) = 1;
x3 = x3(1:2,:) + randn(size(x3(1:2,:)))*scale; x3(3,:) = 1;
//  Homography estimation for each image
XY = [X'; Y'; ones(size(X'))];
H = cell(1,3);
H{1} = estimate_homography(XY,x1,width,height);
H{2} = estimate_homography(XY,x2,width,height);
H{3} = estimate_homography(XY,x3,width,height);
//  Refine homography by LM
VM = [];
for i=1:3
   h = H{i};
//  Intrinsic parameters estimation
   v11 = [h(1,1)*h(1,1), h(1,1)*h(2,1)+h(2,1)*h(1,1), h(2,1)*h(2,1), ...
   h(3,1)*h(1,1)+h(1,1)*h(3,1), h(3,1)*h(2,1)+h(2,1)*h(3,1) h(3,1)*h(3,1)];
   v22 = [h(1,2)*h(1,2), h(1,2)*h(2,2)+h(2,2)*h(1,2), h(2,2)*h(2,2), ...
   h(3,2)*h(1,2)+h(1,2)*h(3,2), h(3,2)*h(2,2)+h(2,2)*h(3,2) h(3,2)*h(3,2)];
   v12 = [h(1,1)*h(1,2), h(1,1)*h(2,2)+h(2,1)*h(1,2), h(2,1)*h(2,2), ...
   h(3,1)*h(1,2)+h(1,1)*h(3,2), h(3,1)*h(2,2)+h(2,1)*h(3,2) h(3,1)*h(3,2)];
   VM = [VM; v12; v11-v22];
end
[U D V] = svd(VM);
b = V(:,end);
v0 = (b(2)*b(4)-b(1)*b(5)) / (b(1)*b(3)-b(2)^2);
lamda = b(6)-(b(4)^2+v0*(b(2)*b(4)-b(1)*b(5)))/b(1);
alpha = sqrt(lamda/b(1));
beta = sqrt(lamda*b(1)/(b(1)*b(3)-b(2)^2));
c = -b(2)*alpha^2*beta/lamda;
u0 = c*v0/alpha - b(4)*alpha^2/lamda;
K_ = [alpha c u0; 0 beta v0; 0 0 1];

//  Extrinsic parameters estimation
iK_ = inv(K_);
for i=1:3
  
   h = H{i};
   lamda = (1/norm(iK_*h(:,1)) + 1/norm(iK_*h(:,2))) / 2;
   r1 = lamda*iK_*h(:,1);
   r2 = lamda*iK_*h(:,2);
   r3 = cross(r1,r2);
   Rtemp = [r1 r2 r3];
   [U D V] = svd(Rtemp);
   R_{i} = U*V';
   t_{i} = lamda*iK_*h(:,3);
   C = -R_{i}'*t_{i};
   if C(3) < 0
       R_{i}(:,1) = -R_{i}(:,1);
       R_{i}(:,2) = -R_{i}(:,2);
       t_{i} = -t_{i};
   end
end
//  //  Camera projection matrix (P = K*[R t])
P = cell(1,3);
P{1} = K_*[R_{1} -t_{1}];
P{2} = K_*[R_{2} -t_{2}];
P{3} = K_*[R_{3} -t_{3}];
temp = [X'; Y'; Z';ones(1,length(X))];
x1_ = P{1}*temp;
x2_ = P{2}*temp;
x3_ = P{3}*temp;
x1_(1,:) = x1_(1,:)./x1_(3,:); x1_(2,:) = x1_(2,:)./x1_(3,:);
x1_(3,:) = x1_(3,:)./x1_(3,:);
x2_(1,:) = x2_(1,:)./x2_(3,:); x2_(2,:) = x2_(2,:)./x2_(3,:);
x2_(3,:) = x2_(3,:)./x2_(3,:);
x3_(1,:) = x3_(1,:)./x3_(3,:); x3_(2,:) = x3_(2,:)./x3_(3,:);
x3_(3,:) = x3_(3,:)./x1_(3,:);
error_dlt = sum(sum((x1(1:2,:)-x1_(1:2,:)).^2)+sum((x2(1:2,:)-x2_(1:2,:)).^2)+sum((x3(1:2,:)-x3_(1:2,:)).^2));
//  refining
//  //  Camera projection matrix (P = K*[R t])
x = cell(1,3);
x{1} = x1;
x{2} =x2;
x{3} =x3;
P_r = cell(1,3);
for i=1:3
   P{i} = K_*[R_{i} -t_{i}];
   P_r{i} = refined_using_LM(P{i},temp,x{i});
end
x1_ = P_r{1}*temp;
x2_ = P_r{2}*temp;
x3_ = P_r{3}*temp;
x1_(1,:) = x1_(1,:)./x1_(3,:); x1_(2,:) = x1_(2,:)./x1_(3,:);
x1_(3,:) = x1_(3,:)./x1_(3,:);
x2_(1,:) = x2_(1,:)./x2_(3,:); x2_(2,:) = x2_(2,:)./x2_(3,:);
x2_(3,:) = x2_(3,:)./x2_(3,:);
x3_(1,:) = x3_(1,:)./x3_(3,:); x3_(2,:) = x3_(2,:)./x3_(3,:);
x3_(3,:) = x3_(3,:)./x1_(3,:);
error_LM = sum(sum((x1(1:2,:)-x1_(1:2,:)).^2)+sum((x2(1:2,:)-x2_(1:2,:)).^2)+sum((x3(1:2,:)-x3_(1:2,:)).^2));
disp(error_LM>error_dlt); //  comparing lm method error with zheng method. Lm should be less than zheng



