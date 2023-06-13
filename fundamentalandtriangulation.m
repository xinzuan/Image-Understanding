% 1) Generate two synthetic images of 3D points using two virtual cameras.
% The 3D object must not be planar, and two cameras must locate at different locations.
%  image size (pixel)
width = 1024; height = 768;
%  Calibration matrix
K1 = [300-3 0.02 width/2-5; 0 300+2 height/2+10; 0 0 1];
K2 = [300+3 0.01 width/2+5; 0 300-2 height/2-10; 0 0 1];
%  pixel
%  pixel
%  Rotation matrix
thx = -30*pi/180;
thy = -170*pi/180;
thz = -80*pi/180; %  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R1 = Rz*Ry*Rx;
thx = 30*pi/180;
thy = -180*pi/180;
thz = -70*pi/180; %  radian
Rx = [1 0 0; 0 cos(thx) -sin(thx); 0 sin(thx) cos(thx)];
Ry = [cos(thy) 0 sin(thy); 0 1 0; -sin(thy) 0 cos(thy)];
Rz = [cos(thz) -sin(thz) 0; sin(thz) cos(thz) 0; 0 0 1];
R2 = Rz*Ry*Rx;
%  Camera center in world coordinate frame (meter)
C1_ = [0; -3; 10];
C2_ = [-1; 3; 9];
%  Translation vector
t1 = -R1*C1_;
t2 = -R2*C2_;
%  Camera projection matrix (P = K*[R t])
P1 = K1*R1*[eye(3) -C1_];
P2 = K2*R2*[eye(3) -C2_];
%  3D points in world coordinate frame
r = -linspace(0,2) ;
th = linspace(0,2*pi) ;
[R,T] = meshgrid(r,th) ;
X = R.*cos(T) ;
Y = R.*sin(T) ;
Z = R ;
X1 = X(:);
Y1 = Y(:);
Z1 = Z(:);
r = -linspace(0,pi) ;
th = linspace(0,pi) ;
[R,T] = meshgrid(r,th) ;
X = R.*cos(T)  ;
Y = sin(T)+ 5  ;
Z = R/2 ;
X2 = X(:);
Y2 = Y(:);
Z2 = Z(:);
X = [X1;X2];
Y = [Y1;Y2];
Z = [Z1;Z2];
%  Draw 3D points and world coordinate frame
figure; plot3(X1,Y1,Z1,'k.'); hold on;
%  hold on;plot3(X2,Y2,Z2,'m.'); hold on;
hold on; plot3(X(10001:20000),Y(10001:20000),Z(10001:20000),'c.'); hold on;
xlabel('X (meter)');
ylabel('Y (meter)');
zlabel('Z (meter)');
axis equal; grid on;
%  Draw camera coordinate frame
len = 2;
Xp = [len; 0; 0];
Yp = [0; len; 0];
Zp = [0; 0; len];
CX1 = R1'*(Xp - t1); CY1 = R1'*(Yp - t1);
CZ1 = R1'*(Zp - t1);
CX2 = R2'*(Xp - t2); CY2 = R2'*(Yp - t2);
CZ2 = R2'*(Zp - t2);
plot3([C1_(1) CX1(1)],[C1_(2) CX1(2)],[C1_(3) CX1(3)],'b-','LineWidth',2);
plot3([C1_(1) CY1(1)],[C1_(2) CY1(2)],[C1_(3) CY1(3)],'g-','LineWidth',2);
plot3([C1_(1) CZ1(1)],[C1_(2) CZ1(2)],[C1_(3) CZ1(3)],'r-','LineWidth',2);
plot3([C2_(1) CX2(1)],[C2_(2) CX2(2)],[C2_(3) CX2(3)],'b:','LineWidth',2);
plot3([C2_(1) CY2(1)],[C2_(2) CY2(2)],[C2_(3) CY2(3)],'g:','LineWidth',2);
plot3([C2_(1) CZ2(1)],[C2_(2) CZ2(2)],[C2_(3) CZ2(3)],'r:','LineWidth',2);
axis equal; hold off;
%  Acquire images of 3D points
x1 = P1*[X'; Y'; Z'; ones(1,length(X))];
x1(1,:) = x1(1,:)./x1(3,:);
x1(2,:) = x1(2,:)./x1(3,:);
x1(3,:) = x1(3,:)./x1(3,:);
x2 = P2*[X'; Y'; Z'; ones(1,length(X))];
x2(1,:) = x2(1,:)./x2(3,:);
x2(2,:) = x2(2,:)./x2(3,:);
x2(3,:) = x2(3,:)./x2(3,:);
%  Draw images of 3D points
figure;
subplot(121); plot(x1(1,:),x1(2,:),'k.'); hold on;
plot(x1(1,10001:20000),x1(2,10001:20000),'c.'); axis equal;
axis([1 width 1 height]);
xlabel('x (pixel)');
hold off;
subplot(122); plot(x2(1,:),x2(2,:),'k.'); hold on;
plot(x2(1,10001:20000),x2(2,10001:20000),'c.'); axis equal;
axis([1 width 1 height]);
xlabel('x (pixel)');
hold off;
save('data_twoviews.mat','K1','R1','t1','K2','R2','t2','X','Y','Z','x1','x2','width','height');

% 2) Add Gaussian noise to the 2D image points.
% 3) Add some outliers to the noise-contaminated 2D points.


%  Load data
load('data_twoviews.mat');
%  Add noise
scale =2;
x1o = x1(1:2,:);
x2o = x2(1:2,:);
rng(1,'philox')
x1 = x1(1:2,:) + randn(size(x1(1:2,:)))*scale;
x2 = x2(1:2,:) + randn(size(x2(1:2,:)))*scale;
x1(3,:) = 1;
x2(3,:) = 1;
%  Outliers
rng(1,'philox')
outNum = 20;
xo = (randn(3,outNum)+1.5)*400;
yo = (randn(3,outNum)+1.5)*200;
xo(3,:) = 0.0010;
yo(3,:) = 0.0010;
x1out = [x1 xo];
y1out = [x2 yo];
figure;
subplot(121); plot(x1(1,:),x1(2,:),'r.'); hold on;
plot(x1(1,10001:20000),x1(2,10001:20000),'m.'); axis equal;
subplot(121); plot(x1o(1,:),x1o(2,:),'k.'); hold on;
plot(x1o(1,10001:20000),x1o(2,10001:20000),'c.'); hold on;
plot(xo(1,:),xo(2,:),'g.');axis equal;
axis([1 width 1 height]);
xlabel('x (pixel)');
hold off;
subplot(122); plot(x2(1,:),x2(2,:),'r.'); hold on;
plot(x2(1,10001:20000),x2(2,10001:20000),'m.'); axis equal;
subplot(122); plot(x2o(1,:),x2o(2,:),'k.'); hold on;
plot(x2o(1,10001:20000),x2o(2,10001:20000),'c.'); hold on;
plot(yo(1,:),yo(2,:),'g.');axis equal;
axis([1 width 1 height]);
xlabel('x (pixel)');
hold off;



% 4) Estimate the fundamental matrix using RANSAC and check the result by drawing epipolar lines.

%  Load data
load('data_twoviews.mat');
%  Add noise
scale =0.0;
x1o = x1(1:2,:);
x2o = x2(1:2,:);
rng(1,'philox')
x1 = x1(1:2,:) + randn(size(x1(1:2,:)))*scale;
x2 = x2(1:2,:) + randn(size(x2(1:2,:)))*scale;
x1(3,:) = 1;
x2(3,:) = 1;
%  Outliers
rng(1,'philox')
outNum = 20;
xo = (randn(3,outNum)+1.5)*400;
yo = (randn(3,outNum)+1.5)*200;
xo(3,:) = 0.0010;
yo(3,:) = 0.0010;

x1out = [x1 xo];
y1out = [x2 yo];
%  x1out = x1;
%  y1out = x2;
figure;
subplot(121); plot(x1(1,:),x1(2,:),'r.'); hold on;
plot(x1(1,10001:20000),x1(2,10001:20000),'m.'); axis equal;
subplot(121); plot(x1o(1,:),x1o(2,:),'k.'); hold on;
plot(x1o(1,10001:20000),x1o(2,10001:20000),'c.'); hold on;
plot(xo(1,:),xo(2,:),'g.');axis equal;
axis([1 width 1 height]);
xlabel('x (pixel)');
hold off;
subplot(122); plot(x2(1,:),x2(2,:),'r.'); hold on;
plot(x2(1,10001:20000),x2(2,10001:20000),'m.'); axis equal;
subplot(122); plot(x2o(1,:),x2o(2,:),'k.'); hold on;
plot(x2o(1,10001:20000),x2o(2,10001:20000),'c.'); hold on;
plot(yo(1,:),yo(2,:),'g.');axis equal;
axis([1 width 1 height]);
xlabel('x (pixel)');
hold off;
%  Normalize data
iT1 = [width+height 0 width/2; 0 width+height height/2; 0 0 1];
T1 = inv(iT1);
T2=T1;
x1nmo = T1*x1out;
y2nmo = T2*y1out;
N = 100;
s = 8;  %  Minimum number of points
threshold = 0.3;
[row, column] = size(x1nmo);
numInliersEachIteration = zeros(N,1);
storedModels = {};%zeros(N,3,3);
maxInliner =0;
xa = [x1out ones(size(x1out,1),1)];
xb = [y1out ones(size(y1out,1),1)];
F = zeros(3,3);
for i = 1 : N
  
   %  random subset of points
   subsetIndices = randsample(column, s);
   %  normalize
   x_subset = T1* x1out(:,subsetIndices);
   y_subset = T2* y1out(:,subsetIndices);
      
    Ftemp = estimate_fundamental_matrix(x_subset, y_subset);
  
   Ftemp = T2'*Ftemp*T1;
   eval = diag(xb' * Ftemp * xa);
  
   %  record the number of inliers
   numInlier= size( find(abs(eval) < threshold) , 1);
   if numInlier > maxInliner
  
       maxInliner = numInlier;
       F = Ftemp;
      
   end
end

%  Draw epipolar lines
id = [10000 20000]; %  randomly select point and draw 4 epipole
for i=1:length(id)
    x = x1out(:,id(i));
    l_ = F*x;
    p1 = [0 -l_(3)/l_(2)];
    p2 = [1024 -(1024*l_(1)+l_(3))/l_(2)];
    subplot(121); hold on; plot(x(1),x(2),'bo');
    subplot(122); hold on; plot([p1(1) p2(1)],[p1(2) p2(2)],'b-');
    x_ = y1out(:,id(i));
    l = F'*x_;
    p1 = [0 -l(3)/l(2)]; p2 = [1024 -(1024*l(1)+l(3))/l(2)];
    subplot(122); hold on; plot(x_(1),x_(2),'bo');
    subplot(121); hold on; plot([p1(1) p2(1)],[p1(2) p2(2)],'b-');
end
save('fundamental.mat', 'F','x1out','y1out','x1','x2');








% 5) Calculate a rotation matrix and a translation vector using two known calibration matrices.


%  Load data
load('data_twoviews.mat');
x1wn = x1;
x2wn = x2;
load('fundamental.mat')
%  Rotation and translation calculation
E = K2'*F*K1;
[U D V] = svd(E);
W = [0 -1 0; 1 0 0; 0 0 1];
Rc1 = U*W*V'; Rc2 = U*W'*V';
tc1 = U(:,3); tc2 = -U(:,3);
%  Triangulation for selecting R and t combination
P1 = K1*[eye(3) zeros(3,1)];
P2 = cell(1,4);
t = cell(1,4);
R = cell(1,4);
P2{1} = K2*[Rc1 tc1]; R{1} = Rc1; t{1} = tc1;
P2{2} = K2*[Rc1 tc2]; R{2} = Rc1; t{2} = tc2;
P2{3} = K2*[Rc2 tc1]; R{3} = Rc2; t{3} = tc1;
P2{4} = K2*[Rc2 tc2]; R{4} = Rc2; t{4} = tc2;
x1 = x1out;
x2=y1out;
for i=1:4
   A = [x1(1,1)*P1(3,:) - P1(1,:);
   x1(2,1)*P1(3,:) - P1(2,:);
   x2(1,1)*P2{i}(3,:) - P2{i}(1,:);
   x2(2,1)*P2{i}(3,:) - P2{i}(2,:)];
   [U D V] = svd(A);
   X = V(:,end); pointZ(i) = X(3)./X(4);
   a = R{i}'*[0;0;1]; axisZ(i) = a(3);
end
id = find(pointZ > 0 & sign(axisZ)==sign(pointZ));
R = R{id}; t = t{id};
%  Triangulation of all points
P1 = K1*[eye(3) zeros(3,1)];
P2 = K2*[R t];
for i=1:length(x1)
   A = [x1(1,i)*P1(3,:) - P1(1,:);
   x1(2,i)*P1(3,:) - P1(2,:);
   x2(1,i)*P2(3,:) - P2(1,:);
   x2(2,i)*P2(3,:) - P2(2,:)];
   [U D V] = svd(A);
   X = V(:,end);
   XYZ(:,i) = X./X(4);
end
figure; plot3(XYZ(1,:),XYZ(2,:),XYZ(3,:),'k.');
hold on; plot3(XYZ(1,10001:20000),XYZ(2,10001:20000),XYZ(3,10001:20000),'c.');
xlabel('X'); ylabel('Y'); zlabel('Z'); axis equal; grid on;




% 6) Reconstruct the 3D structure of the corresponding points with the known baseline
%  Metric recovery
C1 = -R1*t1;
C2 = -R2*t2;
baseline = sqrt(sum((C1-C2).^2));
XYZ = XYZ*baseline;
figure; plot3(XYZ(1,:),XYZ(2,:),XYZ(3,:),'k.');
hold on; plot3(XYZ(1,10001:20000),XYZ(2,10001:20000),XYZ(3,10001:20000),'c.');
xlabel('X (meter)'); ylabel('Y (meter)'); zlabel('Z (meter)');
axis equal; grid on;
