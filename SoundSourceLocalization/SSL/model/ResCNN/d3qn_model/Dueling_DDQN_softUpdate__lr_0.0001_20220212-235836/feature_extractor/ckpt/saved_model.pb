??1
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??'

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:?*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:?@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ԟ
valueɟBş B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10

signatures
#_self_saveable_object_factories
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
%
#_self_saveable_object_factories
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
?
 layer_with_weights-0
 layer-0
!layer_with_weights-1
!layer-1
"layer-2
#layer_with_weights-2
#layer-3
$layer_with_weights-3
$layer-4
%layer-5
&layer_with_weights-4
&layer-6
'layer_with_weights-5
'layer-7
#(_self_saveable_object_factories
)	variables
*trainable_variables
+regularization_losses
,	keras_api
4
#-_self_saveable_object_factories
.	keras_api
w
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6layer-2
7layer_with_weights-2
7layer-3
8layer_with_weights-3
8layer-4
9layer-5
:layer_with_weights-4
:layer-6
;layer_with_weights-5
;layer-7
#<_self_saveable_object_factories
=	variables
>trainable_variables
?regularization_losses
@	keras_api
4
#A_self_saveable_object_factories
B	keras_api
w
#C_self_saveable_object_factories
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
#K_self_saveable_object_factories
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
4
#P_self_saveable_object_factories
Q	keras_api
w
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
 
 
 
 
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
 
 
?
?metrics
	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
trainable_variables
regularization_losses
 
?

Wkernel
Xbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	Ygamma
Zbeta
[moving_mean
\moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?

]kernel
^bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	_gamma
`beta
amoving_mean
bmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
V
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
 
 
?
?metrics
	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
trainable_variables
regularization_losses
?

ckernel
dbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?

ikernel
jbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	kgamma
lbeta
mmoving_mean
nmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?

okernel
pbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	qgamma
rbeta
smoving_mean
tmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
 
 
?
?metrics
)	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
*trainable_variables
+regularization_losses
 
 
 
 
 
 
?
?metrics
0	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
1trainable_variables
2regularization_losses
?

ukernel
vbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	wgamma
xbeta
ymoving_mean
zmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?

{kernel
|bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17
 
 
?
?metrics
=	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
>trainable_variables
?regularization_losses
 
 
 
 
 
 
?
?metrics
D	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
Etrainable_variables
Fregularization_losses
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
0
?0
?1
?2
?3
?4
?5
 
 
?
?metrics
L	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
Mtrainable_variables
Nregularization_losses
 
 
 
 
 
 
?
?metrics
S	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
Ttrainable_variables
Uregularization_losses
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_4/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_4/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_4/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_4/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_5/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_5/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_5/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_5/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_6/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_6/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_6/gamma'variables/38/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_6/beta'variables/39/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_6/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_6/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_7/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_7/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_7/gamma'variables/44/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_7/beta'variables/45/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/47/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_8/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_8/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_8/gamma'variables/50/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_8/beta'variables/51/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/52/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/53/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
N
0
1
2
3
4
5
6
7
	8

9
10
 
 

W0
X1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

Y0
Z1
[2
\3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 

]0
^1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

_0
`1
a2
b3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
V
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
*
0
1
2
3
4
5
 
 

c0
d1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

e0
f1
g2
h3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 

i0
j1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

k0
l1
m2
n3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 

o0
p1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

q0
r1
s2
t3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
8
 0
!1
"2
#3
$4
%5
&6
'7
 
 
 
 
 
 
 

u0
v1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

w0
x1
y2
z3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 

{0
|1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 

}0
~1
2
?3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 

?0
?1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
?0
?1
?2
?3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17
8
40
51
62
73
84
95
:6
;7
 
 
 
 
 
 
 

?0
?1
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
?0
?1
?2
?3
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
 
 
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
 
 
0
?0
?1
?2
?3
?4
?5

H0
I1
J2
 
 
 
 
 
 
 
 

W0
X1
 
 
 
 

Y0
Z1
[2
\3
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 

_0
`1
a2
b3
 
 
 
 
 
 
 
 
 

c0
d1
 
 
 
 

e0
f1
g2
h3
 
 
 
 
 
 
 
 
 

i0
j1
 
 
 
 

k0
l1
m2
n3
 
 
 
 
 
 
 
 
 

o0
p1
 
 
 
 

q0
r1
s2
t3
 
 
 
 

u0
v1
 
 
 
 

w0
x1
y2
z3
 
 
 
 
 
 
 
 
 

{0
|1
 
 
 
 

}0
~1
2
?3
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
?0
?1
?2
?3
 
 
 
 

?0
?1
 
 
 
 
 
?0
?1
?2
?3
 
 
 
 
 
 
 
?
serving_default_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_43656839
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOpConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_43659692
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_43659864??$
?V
?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43657965

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
activation_4/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
activation_5/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_5/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_6_layer_call_fn_43659188

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436550142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43655464

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_layer_call_and_return_conditional_losses_43653428

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43653387

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_4_layer_call_fn_43658905

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436542932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
E__inference_feature_layer_call_and_return_conditional_losses_43655640
conv2d_8_input
conv2d_8_43655624
conv2d_8_43655626"
batch_normalization_8_43655629"
batch_normalization_8_43655631"
batch_normalization_8_43655633"
batch_normalization_8_43655635
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_43655624conv2d_8_43655626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_436555202"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_43655629batch_normalization_8_43655631batch_normalization_8_43655633batch_normalization_8_43655635*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436555712/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_436556122
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659328

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43653797

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43654073

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_43658371

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436532802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659266

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43653479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_43659364

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_8_layer_call_fn_43659484

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436554642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43653311

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43654275

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43654615

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?V
?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43657899

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
activation_4/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
activation_5/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_5/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658493

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_8_layer_call_fn_43659373

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_436555202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659453

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
-__inference_input_conv_layer_call_fn_43653708
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43655014

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_3_layer_call_fn_43658739

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436541652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43654165

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43655032

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43654293

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659157

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_43658293

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436533872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_43659864
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance&
"assignvariableop_6_conv2d_1_kernel$
 assignvariableop_7_conv2d_1_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance'
#assignvariableop_12_conv2d_2_kernel%
!assignvariableop_13_conv2d_2_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance'
#assignvariableop_18_conv2d_3_kernel%
!assignvariableop_19_conv2d_3_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance'
#assignvariableop_24_conv2d_4_kernel%
!assignvariableop_25_conv2d_4_bias3
/assignvariableop_26_batch_normalization_4_gamma2
.assignvariableop_27_batch_normalization_4_beta9
5assignvariableop_28_batch_normalization_4_moving_mean=
9assignvariableop_29_batch_normalization_4_moving_variance'
#assignvariableop_30_conv2d_5_kernel%
!assignvariableop_31_conv2d_5_bias3
/assignvariableop_32_batch_normalization_5_gamma2
.assignvariableop_33_batch_normalization_5_beta9
5assignvariableop_34_batch_normalization_5_moving_mean=
9assignvariableop_35_batch_normalization_5_moving_variance'
#assignvariableop_36_conv2d_6_kernel%
!assignvariableop_37_conv2d_6_bias3
/assignvariableop_38_batch_normalization_6_gamma2
.assignvariableop_39_batch_normalization_6_beta9
5assignvariableop_40_batch_normalization_6_moving_mean=
9assignvariableop_41_batch_normalization_6_moving_variance'
#assignvariableop_42_conv2d_7_kernel%
!assignvariableop_43_conv2d_7_bias3
/assignvariableop_44_batch_normalization_7_gamma2
.assignvariableop_45_batch_normalization_7_beta9
5assignvariableop_46_batch_normalization_7_moving_mean=
9assignvariableop_47_batch_normalization_7_moving_variance'
#assignvariableop_48_conv2d_8_kernel%
!assignvariableop_49_conv2d_8_bias3
/assignvariableop_50_batch_normalization_8_gamma2
.assignvariableop_51_batch_normalization_8_beta9
5assignvariableop_52_batch_normalization_8_moving_mean=
9assignvariableop_53_batch_normalization_8_moving_variance
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
9272
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_5_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_5_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_6_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_6_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_7_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_conv2d_7_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_7_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp.assignvariableop_45_batch_normalization_7_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp5assignvariableop_46_batch_normalization_7_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_normalization_7_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp#assignvariableop_48_conv2d_8_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp!assignvariableop_49_conv2d_8_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp/assignvariableop_50_batch_normalization_8_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp.assignvariableop_51_batch_normalization_8_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp5assignvariableop_52_batch_normalization_8_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp9assignvariableop_53_batch_normalization_8_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54?	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43654055

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_43654242

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_43655520

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
f
J__inference_activation_6_layer_call_and_return_conditional_losses_43655612

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43654646

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_feature_layer_call_fn_43655713
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?	
?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_43655091

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43655553

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
h
L__inference_feature_linear_layer_call_and_return_conditional_losses_43658145

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
E__inference_feature_layer_call_and_return_conditional_losses_43658107

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
activation_6/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
activation_6/Relu?
IdentityIdentityactivation_6/Relu:activations:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
.__inference_res_block_0_layer_call_fn_43654468
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436544292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43653966

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_43654022

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_7_layer_call_fn_43659279

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436551242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_layer_call_and_return_conditional_losses_43658160

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_res_block_1_layer_call_fn_43655406
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436553672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?.
?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655179
conv2d_5_input
conv2d_5_43654882
conv2d_5_43654884"
batch_normalization_5_43654949"
batch_normalization_5_43654951"
batch_normalization_5_43654953"
batch_normalization_5_43654955
conv2d_6_43654992
conv2d_6_43654994"
batch_normalization_6_43655059"
batch_normalization_6_43655061"
batch_normalization_6_43655063"
batch_normalization_6_43655065
conv2d_7_43655102
conv2d_7_43655104"
batch_normalization_7_43655169"
batch_normalization_7_43655171"
batch_normalization_7_43655173"
batch_normalization_7_43655175
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_43654882conv2d_5_43654884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_436548712"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_43654949batch_normalization_5_43654951batch_normalization_5_43654953batch_normalization_5_43654955*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436549042/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_436549632
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_43654992conv2d_6_43654994*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_436549812"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_43655059batch_normalization_6_43655061batch_normalization_6_43655063batch_normalization_6_43655065*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436550142/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_436550732
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_43655102conv2d_7_43655104*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_436550912"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_43655169batch_normalization_7_43655171batch_normalization_7_43655173batch_normalization_7_43655175*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436551242/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?
M
1__inference_feature_linear_layer_call_fn_43658150

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_436561192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658726

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_6_layer_call_fn_43659139

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436547462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659409

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
f
J__inference_activation_6_layer_call_and_return_conditional_losses_43659502

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43659004

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658664

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_5_layer_call_fn_43658986

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436546462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43655571

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
f
J__inference_activation_1_layer_call_and_return_conditional_losses_43658451

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
`
D__inference_relu_1_layer_call_and_return_conditional_losses_43658052

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659175

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?.
?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655227
conv2d_5_input
conv2d_5_43655182
conv2d_5_43655184"
batch_normalization_5_43655187"
batch_normalization_5_43655189"
batch_normalization_5_43655191"
batch_normalization_5_43655193
conv2d_6_43655197
conv2d_6_43655199"
batch_normalization_6_43655202"
batch_normalization_6_43655204"
batch_normalization_6_43655206"
batch_normalization_6_43655208
conv2d_7_43655212
conv2d_7_43655214"
batch_normalization_7_43655217"
batch_normalization_7_43655219"
batch_normalization_7_43655221"
batch_normalization_7_43655223
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_43655182conv2d_5_43655184*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_436548712"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_43655187batch_normalization_5_43655189batch_normalization_5_43655191batch_normalization_5_43655193*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436549222/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_436549632
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_43655197conv2d_6_43655199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_436549812"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_43655202batch_normalization_6_43655204batch_normalization_6_43655206batch_normalization_6_43655208*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436550322/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_436550732
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_43655212conv2d_7_43655214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_436550912"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_43655217batch_normalization_7_43655219batch_normalization_7_43655221batch_normalization_7_43655223*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436551422/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?:
?
C__inference_model_layer_call_and_return_conditional_losses_43656251	
input
input_conv_43656131
input_conv_43656133
input_conv_43656135
input_conv_43656137
input_conv_43656139
input_conv_43656141
input_conv_43656143
input_conv_43656145
input_conv_43656147
input_conv_43656149
input_conv_43656151
input_conv_43656153
res_block_0_43656156
res_block_0_43656158
res_block_0_43656160
res_block_0_43656162
res_block_0_43656164
res_block_0_43656166
res_block_0_43656168
res_block_0_43656170
res_block_0_43656172
res_block_0_43656174
res_block_0_43656176
res_block_0_43656178
res_block_0_43656180
res_block_0_43656182
res_block_0_43656184
res_block_0_43656186
res_block_0_43656188
res_block_0_43656190
res_block_1_43656195
res_block_1_43656197
res_block_1_43656199
res_block_1_43656201
res_block_1_43656203
res_block_1_43656205
res_block_1_43656207
res_block_1_43656209
res_block_1_43656211
res_block_1_43656213
res_block_1_43656215
res_block_1_43656217
res_block_1_43656219
res_block_1_43656221
res_block_1_43656223
res_block_1_43656225
res_block_1_43656227
res_block_1_43656229
feature_43656234
feature_43656236
feature_43656238
feature_43656240
feature_43656242
feature_43656244
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_43656131input_conv_43656133input_conv_43656135input_conv_43656137input_conv_43656139input_conv_43656141input_conv_43656143input_conv_43656145input_conv_43656147input_conv_43656149input_conv_43656151input_conv_43656153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536812$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_43656156res_block_0_43656158res_block_0_43656160res_block_0_43656162res_block_0_43656164res_block_0_43656166res_block_0_43656168res_block_0_43656170res_block_0_43656172res_block_0_43656174res_block_0_43656176res_block_0_43656178res_block_0_43656180res_block_0_43656182res_block_0_43656184res_block_0_43656186res_block_0_43656188res_block_0_43656190*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436545182%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_436559252
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_43656195res_block_1_43656197res_block_1_43656199res_block_1_43656201res_block_1_43656203res_block_1_43656205res_block_1_43656207res_block_1_43656209res_block_1_43656211res_block_1_43656213res_block_1_43656215res_block_1_43656217res_block_1_43656219res_block_1_43656221res_block_1_43656223res_block_1_43656225res_block_1_43656227res_block_1_43656229*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436553672%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_436560582
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_43656234feature_43656236feature_43656238feature_43656240feature_43656242feature_43656244*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556982!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_436561192 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
`
D__inference_relu_0_layer_call_and_return_conditional_losses_43655925

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_43659068

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_6_layer_call_fn_43659077

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_436549812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_3_layer_call_fn_43658690

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436538972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_43654981

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43653180

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_43658619

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
`
D__inference_relu_0_layer_call_and_return_conditional_losses_43657828

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
K
/__inference_activation_5_layer_call_fn_43659211

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_436550732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43654746

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?3
C__inference_model_layer_call_and_return_conditional_losses_43657036

inputs4
0input_conv_conv2d_conv2d_readvariableop_resource5
1input_conv_conv2d_biasadd_readvariableop_resource:
6input_conv_batch_normalization_readvariableop_resource<
8input_conv_batch_normalization_readvariableop_1_resourceK
Ginput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Iinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2input_conv_conv2d_1_conv2d_readvariableop_resource7
3input_conv_conv2d_1_biasadd_readvariableop_resource<
8input_conv_batch_normalization_1_readvariableop_resource>
:input_conv_batch_normalization_1_readvariableop_1_resourceM
Iinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Kinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_2_conv2d_readvariableop_resource8
4res_block_0_conv2d_2_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_2_readvariableop_resource?
;res_block_0_batch_normalization_2_readvariableop_1_resourceN
Jres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_3_conv2d_readvariableop_resource8
4res_block_0_conv2d_3_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_3_readvariableop_resource?
;res_block_0_batch_normalization_3_readvariableop_1_resourceN
Jres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_4_conv2d_readvariableop_resource8
4res_block_0_conv2d_4_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_4_readvariableop_resource?
;res_block_0_batch_normalization_4_readvariableop_1_resourceN
Jres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_5_conv2d_readvariableop_resource8
4res_block_1_conv2d_5_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_5_readvariableop_resource?
;res_block_1_batch_normalization_5_readvariableop_1_resourceN
Jres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_6_conv2d_readvariableop_resource8
4res_block_1_conv2d_6_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_6_readvariableop_resource?
;res_block_1_batch_normalization_6_readvariableop_1_resourceN
Jres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_7_conv2d_readvariableop_resource8
4res_block_1_conv2d_7_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_7_readvariableop_resource?
;res_block_1_batch_normalization_7_readvariableop_1_resourceN
Jres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource3
/feature_conv2d_8_conv2d_readvariableop_resource4
0feature_conv2d_8_biasadd_readvariableop_resource9
5feature_batch_normalization_8_readvariableop_resource;
7feature_batch_normalization_8_readvariableop_1_resourceJ
Ffeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceL
Hfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?,feature/batch_normalization_8/ReadVariableOp?.feature/batch_normalization_8/ReadVariableOp_1?'feature/conv2d_8/BiasAdd/ReadVariableOp?&feature/conv2d_8/Conv2D/ReadVariableOp?>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-input_conv/batch_normalization/ReadVariableOp?/input_conv/batch_normalization/ReadVariableOp_1?@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/input_conv/batch_normalization_1/ReadVariableOp?1input_conv/batch_normalization_1/ReadVariableOp_1?(input_conv/conv2d/BiasAdd/ReadVariableOp?'input_conv/conv2d/Conv2D/ReadVariableOp?*input_conv/conv2d_1/BiasAdd/ReadVariableOp?)input_conv/conv2d_1/Conv2D/ReadVariableOp?Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_2/ReadVariableOp?2res_block_0/batch_normalization_2/ReadVariableOp_1?Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_3/ReadVariableOp?2res_block_0/batch_normalization_3/ReadVariableOp_1?Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_4/ReadVariableOp?2res_block_0/batch_normalization_4/ReadVariableOp_1?+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?*res_block_0/conv2d_2/Conv2D/ReadVariableOp?+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?*res_block_0/conv2d_3/Conv2D/ReadVariableOp?+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?*res_block_0/conv2d_4/Conv2D/ReadVariableOp?Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_5/ReadVariableOp?2res_block_1/batch_normalization_5/ReadVariableOp_1?Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_6/ReadVariableOp?2res_block_1/batch_normalization_6/ReadVariableOp_1?Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_7/ReadVariableOp?2res_block_1/batch_normalization_7/ReadVariableOp_1?+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?*res_block_1/conv2d_5/Conv2D/ReadVariableOp?+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?*res_block_1/conv2d_6/Conv2D/ReadVariableOp?+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
'input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp0input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'input_conv/conv2d/Conv2D/ReadVariableOp?
input_conv/conv2d/Conv2DConv2Dinputs/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
input_conv/conv2d/Conv2D?
(input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp1input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(input_conv/conv2d/BiasAdd/ReadVariableOp?
input_conv/conv2d/BiasAddBiasAdd!input_conv/conv2d/Conv2D:output:00input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
input_conv/conv2d/BiasAdd?
-input_conv/batch_normalization/ReadVariableOpReadVariableOp6input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-input_conv/batch_normalization/ReadVariableOp?
/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp8input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/input_conv/batch_normalization/ReadVariableOp_1?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3"input_conv/conv2d/BiasAdd:output:05input_conv/batch_normalization/ReadVariableOp:value:07input_conv/batch_normalization/ReadVariableOp_1:value:0Finput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hinput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/input_conv/batch_normalization/FusedBatchNormV3?
input_conv/activation/ReluRelu3input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
input_conv/activation/Relu?
)input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02+
)input_conv/conv2d_1/Conv2D/ReadVariableOp?
input_conv/conv2d_1/Conv2DConv2D(input_conv/activation/Relu:activations:01input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
input_conv/conv2d_1/Conv2D?
*input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*input_conv/conv2d_1/BiasAdd/ReadVariableOp?
input_conv/conv2d_1/BiasAddBiasAdd#input_conv/conv2d_1/Conv2D:output:02input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
input_conv/conv2d_1/BiasAdd?
/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp8input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/input_conv/batch_normalization_1/ReadVariableOp?
1input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp:input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1input_conv/batch_normalization_1/ReadVariableOp_1?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
1input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$input_conv/conv2d_1/BiasAdd:output:07input_conv/batch_normalization_1/ReadVariableOp:value:09input_conv/batch_normalization_1/ReadVariableOp_1:value:0Hinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 23
1input_conv/batch_normalization_1/FusedBatchNormV3?
input_conv/activation_1/ReluRelu5input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
input_conv/activation_1/Relu?
*res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_2/Conv2D/ReadVariableOp?
res_block_0/conv2d_2/Conv2DConv2D*input_conv/activation_1/Relu:activations:02res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_2/Conv2D?
+res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?
res_block_0/conv2d_2/BiasAddBiasAdd$res_block_0/conv2d_2/Conv2D:output:03res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_2/BiasAdd?
0res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_2/ReadVariableOp?
2res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_2/ReadVariableOp_1?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_2/BiasAdd:output:08res_block_0/batch_normalization_2/ReadVariableOp:value:0:res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_2/FusedBatchNormV3?
res_block_0/activation_2/ReluRelu6res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_2/Relu?
*res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_3/Conv2D/ReadVariableOp?
res_block_0/conv2d_3/Conv2DConv2D+res_block_0/activation_2/Relu:activations:02res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_3/Conv2D?
+res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?
res_block_0/conv2d_3/BiasAddBiasAdd$res_block_0/conv2d_3/Conv2D:output:03res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_3/BiasAdd?
0res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_3/ReadVariableOp?
2res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_3/ReadVariableOp_1?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_3/BiasAdd:output:08res_block_0/batch_normalization_3/ReadVariableOp:value:0:res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_3/FusedBatchNormV3?
res_block_0/activation_3/ReluRelu6res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_3/Relu?
*res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_4/Conv2D/ReadVariableOp?
res_block_0/conv2d_4/Conv2DConv2D+res_block_0/activation_3/Relu:activations:02res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_4/Conv2D?
+res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?
res_block_0/conv2d_4/BiasAddBiasAdd$res_block_0/conv2d_4/Conv2D:output:03res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_4/BiasAdd?
0res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_4/ReadVariableOp?
2res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_4/ReadVariableOp_1?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_4/BiasAdd:output:08res_block_0/batch_normalization_4/ReadVariableOp:value:0:res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_4/FusedBatchNormV3?
tf.__operators__.add/AddV2AddV2*input_conv/activation_1/Relu:activations:06res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2|
relu_0/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_0/Relu?
*res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_5/Conv2D/ReadVariableOp?
res_block_1/conv2d_5/Conv2DConv2Drelu_0/Relu:activations:02res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_5/Conv2D?
+res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?
res_block_1/conv2d_5/BiasAddBiasAdd$res_block_1/conv2d_5/Conv2D:output:03res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_5/BiasAdd?
0res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_5/ReadVariableOp?
2res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_5/ReadVariableOp_1?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_5/BiasAdd:output:08res_block_1/batch_normalization_5/ReadVariableOp:value:0:res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_5/FusedBatchNormV3?
res_block_1/activation_4/ReluRelu6res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_4/Relu?
*res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_6/Conv2D/ReadVariableOp?
res_block_1/conv2d_6/Conv2DConv2D+res_block_1/activation_4/Relu:activations:02res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_6/Conv2D?
+res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?
res_block_1/conv2d_6/BiasAddBiasAdd$res_block_1/conv2d_6/Conv2D:output:03res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_6/BiasAdd?
0res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_6/ReadVariableOp?
2res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_6/ReadVariableOp_1?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_6/BiasAdd:output:08res_block_1/batch_normalization_6/ReadVariableOp:value:0:res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_6/FusedBatchNormV3?
res_block_1/activation_5/ReluRelu6res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_5/Relu?
*res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
res_block_1/conv2d_7/Conv2DConv2D+res_block_1/activation_5/Relu:activations:02res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_7/Conv2D?
+res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?
res_block_1/conv2d_7/BiasAddBiasAdd$res_block_1/conv2d_7/Conv2D:output:03res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_7/BiasAdd?
0res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_7/ReadVariableOp?
2res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_7/ReadVariableOp_1?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_7/BiasAdd:output:08res_block_1/batch_normalization_7/ReadVariableOp:value:0:res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_7/FusedBatchNormV3?
tf.__operators__.add_1/AddV2AddV2relu_0/Relu:activations:06res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2~
relu_1/ReluRelu tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_1/Relu?
&feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&feature/conv2d_8/Conv2D/ReadVariableOp?
feature/conv2d_8/Conv2DConv2Drelu_1/Relu:activations:0.feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
feature/conv2d_8/Conv2D?
'feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'feature/conv2d_8/BiasAdd/ReadVariableOp?
feature/conv2d_8/BiasAddBiasAdd feature/conv2d_8/Conv2D:output:0/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
feature/conv2d_8/BiasAdd?
,feature/batch_normalization_8/ReadVariableOpReadVariableOp5feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02.
,feature/batch_normalization_8/ReadVariableOp?
.feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp7feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype020
.feature/batch_normalization_8/ReadVariableOp_1?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3!feature/conv2d_8/BiasAdd:output:04feature/batch_normalization_8/ReadVariableOp:value:06feature/batch_normalization_8/ReadVariableOp_1:value:0Efeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gfeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 20
.feature/batch_normalization_8/FusedBatchNormV3?
feature/activation_6/ReluRelu2feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
feature/activation_6/Relu?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose'feature/activation_6/Relu:activations:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
IdentityIdentity$tf.compat.v1.transpose/transpose:y:0>^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^feature/batch_normalization_8/ReadVariableOp/^feature/batch_normalization_8/ReadVariableOp_1(^feature/conv2d_8/BiasAdd/ReadVariableOp'^feature/conv2d_8/Conv2D/ReadVariableOp?^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpA^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^input_conv/batch_normalization/ReadVariableOp0^input_conv/batch_normalization/ReadVariableOp_1A^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^input_conv/batch_normalization_1/ReadVariableOp2^input_conv/batch_normalization_1/ReadVariableOp_1)^input_conv/conv2d/BiasAdd/ReadVariableOp(^input_conv/conv2d/Conv2D/ReadVariableOp+^input_conv/conv2d_1/BiasAdd/ReadVariableOp*^input_conv/conv2d_1/Conv2D/ReadVariableOpB^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_2/ReadVariableOp3^res_block_0/batch_normalization_2/ReadVariableOp_1B^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_3/ReadVariableOp3^res_block_0/batch_normalization_3/ReadVariableOp_1B^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_4/ReadVariableOp3^res_block_0/batch_normalization_4/ReadVariableOp_1,^res_block_0/conv2d_2/BiasAdd/ReadVariableOp+^res_block_0/conv2d_2/Conv2D/ReadVariableOp,^res_block_0/conv2d_3/BiasAdd/ReadVariableOp+^res_block_0/conv2d_3/Conv2D/ReadVariableOp,^res_block_0/conv2d_4/BiasAdd/ReadVariableOp+^res_block_0/conv2d_4/Conv2D/ReadVariableOpB^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_5/ReadVariableOp3^res_block_1/batch_normalization_5/ReadVariableOp_1B^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_6/ReadVariableOp3^res_block_1/batch_normalization_6/ReadVariableOp_1B^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_7/ReadVariableOp3^res_block_1/batch_normalization_7/ReadVariableOp_1,^res_block_1/conv2d_5/BiasAdd/ReadVariableOp+^res_block_1/conv2d_5/Conv2D/ReadVariableOp,^res_block_1/conv2d_6/BiasAdd/ReadVariableOp+^res_block_1/conv2d_6/Conv2D/ReadVariableOp,^res_block_1/conv2d_7/BiasAdd/ReadVariableOp+^res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2~
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,feature/batch_normalization_8/ReadVariableOp,feature/batch_normalization_8/ReadVariableOp2`
.feature/batch_normalization_8/ReadVariableOp_1.feature/batch_normalization_8/ReadVariableOp_12R
'feature/conv2d_8/BiasAdd/ReadVariableOp'feature/conv2d_8/BiasAdd/ReadVariableOp2P
&feature/conv2d_8/Conv2D/ReadVariableOp&feature/conv2d_8/Conv2D/ReadVariableOp2?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-input_conv/batch_normalization/ReadVariableOp-input_conv/batch_normalization/ReadVariableOp2b
/input_conv/batch_normalization/ReadVariableOp_1/input_conv/batch_normalization/ReadVariableOp_12?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/input_conv/batch_normalization_1/ReadVariableOp/input_conv/batch_normalization_1/ReadVariableOp2f
1input_conv/batch_normalization_1/ReadVariableOp_11input_conv/batch_normalization_1/ReadVariableOp_12T
(input_conv/conv2d/BiasAdd/ReadVariableOp(input_conv/conv2d/BiasAdd/ReadVariableOp2R
'input_conv/conv2d/Conv2D/ReadVariableOp'input_conv/conv2d/Conv2D/ReadVariableOp2X
*input_conv/conv2d_1/BiasAdd/ReadVariableOp*input_conv/conv2d_1/BiasAdd/ReadVariableOp2V
)input_conv/conv2d_1/Conv2D/ReadVariableOp)input_conv/conv2d_1/Conv2D/ReadVariableOp2?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_2/ReadVariableOp0res_block_0/batch_normalization_2/ReadVariableOp2h
2res_block_0/batch_normalization_2/ReadVariableOp_12res_block_0/batch_normalization_2/ReadVariableOp_12?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_3/ReadVariableOp0res_block_0/batch_normalization_3/ReadVariableOp2h
2res_block_0/batch_normalization_3/ReadVariableOp_12res_block_0/batch_normalization_3/ReadVariableOp_12?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_4/ReadVariableOp0res_block_0/batch_normalization_4/ReadVariableOp2h
2res_block_0/batch_normalization_4/ReadVariableOp_12res_block_0/batch_normalization_4/ReadVariableOp_12Z
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp+res_block_0/conv2d_2/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_2/Conv2D/ReadVariableOp*res_block_0/conv2d_2/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp+res_block_0/conv2d_3/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_3/Conv2D/ReadVariableOp*res_block_0/conv2d_3/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp+res_block_0/conv2d_4/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_4/Conv2D/ReadVariableOp*res_block_0/conv2d_4/Conv2D/ReadVariableOp2?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_5/ReadVariableOp0res_block_1/batch_normalization_5/ReadVariableOp2h
2res_block_1/batch_normalization_5/ReadVariableOp_12res_block_1/batch_normalization_5/ReadVariableOp_12?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_6/ReadVariableOp0res_block_1/batch_normalization_6/ReadVariableOp2h
2res_block_1/batch_normalization_6/ReadVariableOp_12res_block_1/batch_normalization_6/ReadVariableOp_12?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_7/ReadVariableOp0res_block_1/batch_normalization_7/ReadVariableOp2h
2res_block_1/batch_normalization_7/ReadVariableOp_12res_block_1/batch_normalization_7/ReadVariableOp_12Z
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp+res_block_1/conv2d_5/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_5/Conv2D/ReadVariableOp*res_block_1/conv2d_5/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp+res_block_1/conv2d_6/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_6/Conv2D/ReadVariableOp*res_block_1/conv2d_6/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp+res_block_1/conv2d_7/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_7/Conv2D/ReadVariableOp*res_block_1/conv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_43658384

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436533112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_layer_call_fn_43658169

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_436533362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_activation_2_layer_call_and_return_conditional_losses_43658604

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_4_layer_call_fn_43658781

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_436542422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43654183

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
.__inference_res_block_1_layer_call_fn_43658006

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436552782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43653369

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?V
?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43657741

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_3/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_1_layer_call_fn_43658322

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_436534462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_43654132

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_7_layer_call_fn_43659230

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_436550912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43658960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_7_layer_call_fn_43659354

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436548462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?!
?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653581
conv2d_input
conv2d_43653550
conv2d_43653552 
batch_normalization_43653555 
batch_normalization_43653557 
batch_normalization_43653559 
batch_normalization_43653561
conv2d_1_43653565
conv2d_1_43653567"
batch_normalization_1_43653570"
batch_normalization_1_43653572"
batch_normalization_1_43653574"
batch_normalization_1_43653576
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_43653550conv2d_43653552*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_436533362 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_43653555batch_normalization_43653557batch_normalization_43653559batch_normalization_43653561*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436533872-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_436534282
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_43653565conv2d_1_43653567*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_436534462"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_43653570batch_normalization_1_43653572batch_normalization_1_43653574batch_normalization_1_43653576*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436534972/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_436535382
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
?
8__inference_batch_normalization_6_layer_call_fn_43659201

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436550322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_43653446

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658420

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_43658433

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436534792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_1_layer_call_fn_43658446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436534972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_4_layer_call_fn_43658830

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436539662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_feature_layer_call_fn_43655677
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?
?
.__inference_res_block_0_layer_call_fn_43654557
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436545182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?	
?
-__inference_input_conv_layer_call_fn_43653645
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658358

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?:
?	
H__inference_input_conv_layer_call_and_return_conditional_losses_43657551

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
activation/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_1/Relu?
IdentityIdentityactivation_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_5_layer_call_fn_43658924

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_436548712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43653280

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
-__inference_input_conv_layer_call_fn_43657580

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_res_block_0_layer_call_fn_43657823

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436545182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43654846

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
K
/__inference_activation_1_layer_call_fn_43658456

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_436535382
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_43658218

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436531802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43653211

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_feature_layer_call_and_return_conditional_losses_43655621
conv2d_8_input
conv2d_8_43655531
conv2d_8_43655533"
batch_normalization_8_43655598"
batch_normalization_8_43655600"
batch_normalization_8_43655602"
batch_normalization_8_43655604
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_43655531conv2d_8_43655533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_436555202"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_43655598batch_normalization_8_43655600batch_normalization_8_43655602batch_normalization_8_43655604*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436555532/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_436556122
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?
?
&__inference_signature_wrapper_43656839	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_436531222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43654715

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43655142

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_5_layer_call_fn_43659048

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436549222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
E__inference_feature_layer_call_and_return_conditional_losses_43655698

inputs
conv2d_8_43655682
conv2d_8_43655684"
batch_normalization_8_43655687"
batch_normalization_8_43655689"
batch_normalization_8_43655691"
batch_normalization_8_43655693
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_43655682conv2d_8_43655684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_436555202"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_43655687batch_normalization_8_43655689batch_normalization_8_43655691batch_normalization_8_43655693*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436555712/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_436556122
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_7_layer_call_fn_43659292

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436551422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
h
L__inference_feature_linear_layer_call_and_return_conditional_losses_43656119

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658861

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43659022

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_8_layer_call_fn_43659497

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436554952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_activation_3_layer_call_and_return_conditional_losses_43658757

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
.__inference_res_block_1_layer_call_fn_43655317
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436552782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?!
?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653618

inputs
conv2d_43653587
conv2d_43653589 
batch_normalization_43653592 
batch_normalization_43653594 
batch_normalization_43653596 
batch_normalization_43653598
conv2d_1_43653602
conv2d_1_43653604"
batch_normalization_1_43653607"
batch_normalization_1_43653609"
batch_normalization_1_43653611"
batch_normalization_1_43653613
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_43653587conv2d_43653589*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_436533362 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_43653592batch_normalization_43653594batch_normalization_43653596batch_normalization_43653598*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436533692-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_436534282
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_43653602conv2d_1_43653604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_436534462"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_43653607batch_normalization_1_43653609batch_normalization_1_43653611batch_normalization_1_43653613*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436534792/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_436535382
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43653897

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_43659221

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_43658915

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_2_layer_call_fn_43658586

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436537662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43653866

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?.
?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655367

inputs
conv2d_5_43655322
conv2d_5_43655324"
batch_normalization_5_43655327"
batch_normalization_5_43655329"
batch_normalization_5_43655331"
batch_normalization_5_43655333
conv2d_6_43655337
conv2d_6_43655339"
batch_normalization_6_43655342"
batch_normalization_6_43655344"
batch_normalization_6_43655346"
batch_normalization_6_43655348
conv2d_7_43655352
conv2d_7_43655354"
batch_normalization_7_43655357"
batch_normalization_7_43655359"
batch_normalization_7_43655361"
batch_normalization_7_43655363
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_43655322conv2d_5_43655324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_436548712"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_43655327batch_normalization_5_43655329batch_normalization_5_43655331batch_normalization_5_43655333*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436549222/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_436549632
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_43655337conv2d_6_43655339*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_436549812"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_43655342batch_normalization_6_43655344batch_normalization_6_43655346batch_normalization_6_43655348*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436550322/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_436550732
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_43655352conv2d_7_43655354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_436550912"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_43655357batch_normalization_7_43655359batch_normalization_7_43655361batch_normalization_7_43655363*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436551422/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
K
/__inference_activation_4_layer_call_fn_43659058

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_436549632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
*__inference_feature_layer_call_fn_43658124

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?:
?	
H__inference_input_conv_layer_call_and_return_conditional_losses_43657505

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
activation/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_1/Relu?
IdentityIdentityactivation_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_8_layer_call_fn_43659422

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436555532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658402

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_3_layer_call_fn_43658752

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436541832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
I
-__inference_activation_layer_call_fn_43658303

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_436534282
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659391

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43654922

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
d
H__inference_activation_layer_call_and_return_conditional_losses_43658298

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658817

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43654815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43655124

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659310

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658799

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_43657346

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_436563772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658573

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_2_layer_call_fn_43658537

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436540732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?V
?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43657675

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_3/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_43656724	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_436566132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
K
/__inference_activation_3_layer_call_fn_43658762

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_436542242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658708

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
E
)__inference_relu_1_layer_call_fn_43658057

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_436560582
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?:
?
C__inference_model_layer_call_and_return_conditional_losses_43656613

inputs
input_conv_43656493
input_conv_43656495
input_conv_43656497
input_conv_43656499
input_conv_43656501
input_conv_43656503
input_conv_43656505
input_conv_43656507
input_conv_43656509
input_conv_43656511
input_conv_43656513
input_conv_43656515
res_block_0_43656518
res_block_0_43656520
res_block_0_43656522
res_block_0_43656524
res_block_0_43656526
res_block_0_43656528
res_block_0_43656530
res_block_0_43656532
res_block_0_43656534
res_block_0_43656536
res_block_0_43656538
res_block_0_43656540
res_block_0_43656542
res_block_0_43656544
res_block_0_43656546
res_block_0_43656548
res_block_0_43656550
res_block_0_43656552
res_block_1_43656557
res_block_1_43656559
res_block_1_43656561
res_block_1_43656563
res_block_1_43656565
res_block_1_43656567
res_block_1_43656569
res_block_1_43656571
res_block_1_43656573
res_block_1_43656575
res_block_1_43656577
res_block_1_43656579
res_block_1_43656581
res_block_1_43656583
res_block_1_43656585
res_block_1_43656587
res_block_1_43656589
res_block_1_43656591
feature_43656596
feature_43656598
feature_43656600
feature_43656602
feature_43656604
feature_43656606
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_43656493input_conv_43656495input_conv_43656497input_conv_43656499input_conv_43656501input_conv_43656503input_conv_43656505input_conv_43656507input_conv_43656509input_conv_43656511input_conv_43656513input_conv_43656515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536812$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_43656518res_block_0_43656520res_block_0_43656522res_block_0_43656524res_block_0_43656526res_block_0_43656528res_block_0_43656530res_block_0_43656532res_block_0_43656534res_block_0_43656536res_block_0_43656538res_block_0_43656540res_block_0_43656542res_block_0_43656544res_block_0_43656546res_block_0_43656548res_block_0_43656550res_block_0_43656552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436545182%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_436559252
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_43656557res_block_1_43656559res_block_1_43656561res_block_1_43656563res_block_1_43656565res_block_1_43656567res_block_1_43656569res_block_1_43656571res_block_1_43656573res_block_1_43656575res_block_1_43656577res_block_1_43656579res_block_1_43656581res_block_1_43656583res_block_1_43656585res_block_1_43656587res_block_1_43656589res_block_1_43656591*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436553672%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_436560582
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_43656596feature_43656598feature_43656600feature_43656602feature_43656604feature_43656606*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556982!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_436561192 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_43658280

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436533692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_activation_4_layer_call_and_return_conditional_losses_43654963

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
f
J__inference_activation_5_layer_call_and_return_conditional_losses_43655073

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
.__inference_res_block_1_layer_call_fn_43658047

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436553672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_8_layer_call_fn_43659435

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436555712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_4_layer_call_fn_43658843

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436539972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658340

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43653766

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_43658466

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_5_layer_call_fn_43658973

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436546152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_layer_call_and_return_conditional_losses_43653336

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654429

inputs
conv2d_2_43654384
conv2d_2_43654386"
batch_normalization_2_43654389"
batch_normalization_2_43654391"
batch_normalization_2_43654393"
batch_normalization_2_43654395
conv2d_3_43654399
conv2d_3_43654401"
batch_normalization_3_43654404"
batch_normalization_3_43654406"
batch_normalization_3_43654408"
batch_normalization_3_43654410
conv2d_4_43654414
conv2d_4_43654416"
batch_normalization_4_43654419"
batch_normalization_4_43654421"
batch_normalization_4_43654423"
batch_normalization_4_43654425
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_43654384conv2d_2_43654386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_436540222"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_43654389batch_normalization_2_43654391batch_normalization_2_43654393batch_normalization_2_43654395*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436540552/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_436541142
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_43654399conv2d_3_43654401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_436541322"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_43654404batch_normalization_3_43654406batch_normalization_3_43654408batch_normalization_3_43654410*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436541652/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_436542242
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_43654414conv2d_4_43654416*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_436542422"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_43654419batch_normalization_4_43654421batch_normalization_4_43654423batch_normalization_4_43654425*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436542752/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43658942

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_43656488	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_436563772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
8__inference_batch_normalization_2_layer_call_fn_43658599

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436537972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_3_layer_call_fn_43658628

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_436541322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
K
/__inference_activation_6_layer_call_fn_43659507

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_436556122
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
??
?3
C__inference_model_layer_call_and_return_conditional_losses_43657233

inputs4
0input_conv_conv2d_conv2d_readvariableop_resource5
1input_conv_conv2d_biasadd_readvariableop_resource:
6input_conv_batch_normalization_readvariableop_resource<
8input_conv_batch_normalization_readvariableop_1_resourceK
Ginput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Iinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2input_conv_conv2d_1_conv2d_readvariableop_resource7
3input_conv_conv2d_1_biasadd_readvariableop_resource<
8input_conv_batch_normalization_1_readvariableop_resource>
:input_conv_batch_normalization_1_readvariableop_1_resourceM
Iinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Kinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_2_conv2d_readvariableop_resource8
4res_block_0_conv2d_2_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_2_readvariableop_resource?
;res_block_0_batch_normalization_2_readvariableop_1_resourceN
Jres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_3_conv2d_readvariableop_resource8
4res_block_0_conv2d_3_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_3_readvariableop_resource?
;res_block_0_batch_normalization_3_readvariableop_1_resourceN
Jres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_4_conv2d_readvariableop_resource8
4res_block_0_conv2d_4_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_4_readvariableop_resource?
;res_block_0_batch_normalization_4_readvariableop_1_resourceN
Jres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_5_conv2d_readvariableop_resource8
4res_block_1_conv2d_5_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_5_readvariableop_resource?
;res_block_1_batch_normalization_5_readvariableop_1_resourceN
Jres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_6_conv2d_readvariableop_resource8
4res_block_1_conv2d_6_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_6_readvariableop_resource?
;res_block_1_batch_normalization_6_readvariableop_1_resourceN
Jres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_7_conv2d_readvariableop_resource8
4res_block_1_conv2d_7_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_7_readvariableop_resource?
;res_block_1_batch_normalization_7_readvariableop_1_resourceN
Jres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource3
/feature_conv2d_8_conv2d_readvariableop_resource4
0feature_conv2d_8_biasadd_readvariableop_resource9
5feature_batch_normalization_8_readvariableop_resource;
7feature_batch_normalization_8_readvariableop_1_resourceJ
Ffeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceL
Hfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?,feature/batch_normalization_8/ReadVariableOp?.feature/batch_normalization_8/ReadVariableOp_1?'feature/conv2d_8/BiasAdd/ReadVariableOp?&feature/conv2d_8/Conv2D/ReadVariableOp?>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-input_conv/batch_normalization/ReadVariableOp?/input_conv/batch_normalization/ReadVariableOp_1?@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/input_conv/batch_normalization_1/ReadVariableOp?1input_conv/batch_normalization_1/ReadVariableOp_1?(input_conv/conv2d/BiasAdd/ReadVariableOp?'input_conv/conv2d/Conv2D/ReadVariableOp?*input_conv/conv2d_1/BiasAdd/ReadVariableOp?)input_conv/conv2d_1/Conv2D/ReadVariableOp?Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_2/ReadVariableOp?2res_block_0/batch_normalization_2/ReadVariableOp_1?Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_3/ReadVariableOp?2res_block_0/batch_normalization_3/ReadVariableOp_1?Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_4/ReadVariableOp?2res_block_0/batch_normalization_4/ReadVariableOp_1?+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?*res_block_0/conv2d_2/Conv2D/ReadVariableOp?+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?*res_block_0/conv2d_3/Conv2D/ReadVariableOp?+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?*res_block_0/conv2d_4/Conv2D/ReadVariableOp?Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_5/ReadVariableOp?2res_block_1/batch_normalization_5/ReadVariableOp_1?Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_6/ReadVariableOp?2res_block_1/batch_normalization_6/ReadVariableOp_1?Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_7/ReadVariableOp?2res_block_1/batch_normalization_7/ReadVariableOp_1?+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?*res_block_1/conv2d_5/Conv2D/ReadVariableOp?+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?*res_block_1/conv2d_6/Conv2D/ReadVariableOp?+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
'input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp0input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'input_conv/conv2d/Conv2D/ReadVariableOp?
input_conv/conv2d/Conv2DConv2Dinputs/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
input_conv/conv2d/Conv2D?
(input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp1input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(input_conv/conv2d/BiasAdd/ReadVariableOp?
input_conv/conv2d/BiasAddBiasAdd!input_conv/conv2d/Conv2D:output:00input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
input_conv/conv2d/BiasAdd?
-input_conv/batch_normalization/ReadVariableOpReadVariableOp6input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-input_conv/batch_normalization/ReadVariableOp?
/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp8input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/input_conv/batch_normalization/ReadVariableOp_1?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3"input_conv/conv2d/BiasAdd:output:05input_conv/batch_normalization/ReadVariableOp:value:07input_conv/batch_normalization/ReadVariableOp_1:value:0Finput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hinput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/input_conv/batch_normalization/FusedBatchNormV3?
input_conv/activation/ReluRelu3input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
input_conv/activation/Relu?
)input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02+
)input_conv/conv2d_1/Conv2D/ReadVariableOp?
input_conv/conv2d_1/Conv2DConv2D(input_conv/activation/Relu:activations:01input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
input_conv/conv2d_1/Conv2D?
*input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*input_conv/conv2d_1/BiasAdd/ReadVariableOp?
input_conv/conv2d_1/BiasAddBiasAdd#input_conv/conv2d_1/Conv2D:output:02input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
input_conv/conv2d_1/BiasAdd?
/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp8input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/input_conv/batch_normalization_1/ReadVariableOp?
1input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp:input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1input_conv/batch_normalization_1/ReadVariableOp_1?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
1input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$input_conv/conv2d_1/BiasAdd:output:07input_conv/batch_normalization_1/ReadVariableOp:value:09input_conv/batch_normalization_1/ReadVariableOp_1:value:0Hinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 23
1input_conv/batch_normalization_1/FusedBatchNormV3?
input_conv/activation_1/ReluRelu5input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
input_conv/activation_1/Relu?
*res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_2/Conv2D/ReadVariableOp?
res_block_0/conv2d_2/Conv2DConv2D*input_conv/activation_1/Relu:activations:02res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_2/Conv2D?
+res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?
res_block_0/conv2d_2/BiasAddBiasAdd$res_block_0/conv2d_2/Conv2D:output:03res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_2/BiasAdd?
0res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_2/ReadVariableOp?
2res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_2/ReadVariableOp_1?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_2/BiasAdd:output:08res_block_0/batch_normalization_2/ReadVariableOp:value:0:res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_2/FusedBatchNormV3?
res_block_0/activation_2/ReluRelu6res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_2/Relu?
*res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_3/Conv2D/ReadVariableOp?
res_block_0/conv2d_3/Conv2DConv2D+res_block_0/activation_2/Relu:activations:02res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_3/Conv2D?
+res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?
res_block_0/conv2d_3/BiasAddBiasAdd$res_block_0/conv2d_3/Conv2D:output:03res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_3/BiasAdd?
0res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_3/ReadVariableOp?
2res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_3/ReadVariableOp_1?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_3/BiasAdd:output:08res_block_0/batch_normalization_3/ReadVariableOp:value:0:res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_3/FusedBatchNormV3?
res_block_0/activation_3/ReluRelu6res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_3/Relu?
*res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_4/Conv2D/ReadVariableOp?
res_block_0/conv2d_4/Conv2DConv2D+res_block_0/activation_3/Relu:activations:02res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_4/Conv2D?
+res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?
res_block_0/conv2d_4/BiasAddBiasAdd$res_block_0/conv2d_4/Conv2D:output:03res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_4/BiasAdd?
0res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_4/ReadVariableOp?
2res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_4/ReadVariableOp_1?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_4/BiasAdd:output:08res_block_0/batch_normalization_4/ReadVariableOp:value:0:res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_4/FusedBatchNormV3?
tf.__operators__.add/AddV2AddV2*input_conv/activation_1/Relu:activations:06res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2|
relu_0/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_0/Relu?
*res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_5/Conv2D/ReadVariableOp?
res_block_1/conv2d_5/Conv2DConv2Drelu_0/Relu:activations:02res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_5/Conv2D?
+res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?
res_block_1/conv2d_5/BiasAddBiasAdd$res_block_1/conv2d_5/Conv2D:output:03res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_5/BiasAdd?
0res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_5/ReadVariableOp?
2res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_5/ReadVariableOp_1?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_5/BiasAdd:output:08res_block_1/batch_normalization_5/ReadVariableOp:value:0:res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_5/FusedBatchNormV3?
res_block_1/activation_4/ReluRelu6res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_4/Relu?
*res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_6/Conv2D/ReadVariableOp?
res_block_1/conv2d_6/Conv2DConv2D+res_block_1/activation_4/Relu:activations:02res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_6/Conv2D?
+res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?
res_block_1/conv2d_6/BiasAddBiasAdd$res_block_1/conv2d_6/Conv2D:output:03res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_6/BiasAdd?
0res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_6/ReadVariableOp?
2res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_6/ReadVariableOp_1?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_6/BiasAdd:output:08res_block_1/batch_normalization_6/ReadVariableOp:value:0:res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_6/FusedBatchNormV3?
res_block_1/activation_5/ReluRelu6res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_5/Relu?
*res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
res_block_1/conv2d_7/Conv2DConv2D+res_block_1/activation_5/Relu:activations:02res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_7/Conv2D?
+res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?
res_block_1/conv2d_7/BiasAddBiasAdd$res_block_1/conv2d_7/Conv2D:output:03res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_7/BiasAdd?
0res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_7/ReadVariableOp?
2res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_7/ReadVariableOp_1?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_7/BiasAdd:output:08res_block_1/batch_normalization_7/ReadVariableOp:value:0:res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_7/FusedBatchNormV3?
tf.__operators__.add_1/AddV2AddV2relu_0/Relu:activations:06res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2~
relu_1/ReluRelu tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_1/Relu?
&feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&feature/conv2d_8/Conv2D/ReadVariableOp?
feature/conv2d_8/Conv2DConv2Drelu_1/Relu:activations:0.feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
feature/conv2d_8/Conv2D?
'feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'feature/conv2d_8/BiasAdd/ReadVariableOp?
feature/conv2d_8/BiasAddBiasAdd feature/conv2d_8/Conv2D:output:0/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
feature/conv2d_8/BiasAdd?
,feature/batch_normalization_8/ReadVariableOpReadVariableOp5feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02.
,feature/batch_normalization_8/ReadVariableOp?
.feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp7feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype020
.feature/batch_normalization_8/ReadVariableOp_1?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3!feature/conv2d_8/BiasAdd:output:04feature/batch_normalization_8/ReadVariableOp:value:06feature/batch_normalization_8/ReadVariableOp_1:value:0Efeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gfeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 20
.feature/batch_normalization_8/FusedBatchNormV3?
feature/activation_6/ReluRelu2feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
feature/activation_6/Relu?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose'feature/activation_6/Relu:activations:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
IdentityIdentity$tf.compat.v1.transpose/transpose:y:0>^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^feature/batch_normalization_8/ReadVariableOp/^feature/batch_normalization_8/ReadVariableOp_1(^feature/conv2d_8/BiasAdd/ReadVariableOp'^feature/conv2d_8/Conv2D/ReadVariableOp?^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpA^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^input_conv/batch_normalization/ReadVariableOp0^input_conv/batch_normalization/ReadVariableOp_1A^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^input_conv/batch_normalization_1/ReadVariableOp2^input_conv/batch_normalization_1/ReadVariableOp_1)^input_conv/conv2d/BiasAdd/ReadVariableOp(^input_conv/conv2d/Conv2D/ReadVariableOp+^input_conv/conv2d_1/BiasAdd/ReadVariableOp*^input_conv/conv2d_1/Conv2D/ReadVariableOpB^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_2/ReadVariableOp3^res_block_0/batch_normalization_2/ReadVariableOp_1B^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_3/ReadVariableOp3^res_block_0/batch_normalization_3/ReadVariableOp_1B^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_4/ReadVariableOp3^res_block_0/batch_normalization_4/ReadVariableOp_1,^res_block_0/conv2d_2/BiasAdd/ReadVariableOp+^res_block_0/conv2d_2/Conv2D/ReadVariableOp,^res_block_0/conv2d_3/BiasAdd/ReadVariableOp+^res_block_0/conv2d_3/Conv2D/ReadVariableOp,^res_block_0/conv2d_4/BiasAdd/ReadVariableOp+^res_block_0/conv2d_4/Conv2D/ReadVariableOpB^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_5/ReadVariableOp3^res_block_1/batch_normalization_5/ReadVariableOp_1B^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_6/ReadVariableOp3^res_block_1/batch_normalization_6/ReadVariableOp_1B^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_7/ReadVariableOp3^res_block_1/batch_normalization_7/ReadVariableOp_1,^res_block_1/conv2d_5/BiasAdd/ReadVariableOp+^res_block_1/conv2d_5/Conv2D/ReadVariableOp,^res_block_1/conv2d_6/BiasAdd/ReadVariableOp+^res_block_1/conv2d_6/Conv2D/ReadVariableOp,^res_block_1/conv2d_7/BiasAdd/ReadVariableOp+^res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2~
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,feature/batch_normalization_8/ReadVariableOp,feature/batch_normalization_8/ReadVariableOp2`
.feature/batch_normalization_8/ReadVariableOp_1.feature/batch_normalization_8/ReadVariableOp_12R
'feature/conv2d_8/BiasAdd/ReadVariableOp'feature/conv2d_8/BiasAdd/ReadVariableOp2P
&feature/conv2d_8/Conv2D/ReadVariableOp&feature/conv2d_8/Conv2D/ReadVariableOp2?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-input_conv/batch_normalization/ReadVariableOp-input_conv/batch_normalization/ReadVariableOp2b
/input_conv/batch_normalization/ReadVariableOp_1/input_conv/batch_normalization/ReadVariableOp_12?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/input_conv/batch_normalization_1/ReadVariableOp/input_conv/batch_normalization_1/ReadVariableOp2f
1input_conv/batch_normalization_1/ReadVariableOp_11input_conv/batch_normalization_1/ReadVariableOp_12T
(input_conv/conv2d/BiasAdd/ReadVariableOp(input_conv/conv2d/BiasAdd/ReadVariableOp2R
'input_conv/conv2d/Conv2D/ReadVariableOp'input_conv/conv2d/Conv2D/ReadVariableOp2X
*input_conv/conv2d_1/BiasAdd/ReadVariableOp*input_conv/conv2d_1/BiasAdd/ReadVariableOp2V
)input_conv/conv2d_1/Conv2D/ReadVariableOp)input_conv/conv2d_1/Conv2D/ReadVariableOp2?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_2/ReadVariableOp0res_block_0/batch_normalization_2/ReadVariableOp2h
2res_block_0/batch_normalization_2/ReadVariableOp_12res_block_0/batch_normalization_2/ReadVariableOp_12?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_3/ReadVariableOp0res_block_0/batch_normalization_3/ReadVariableOp2h
2res_block_0/batch_normalization_3/ReadVariableOp_12res_block_0/batch_normalization_3/ReadVariableOp_12?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_4/ReadVariableOp0res_block_0/batch_normalization_4/ReadVariableOp2h
2res_block_0/batch_normalization_4/ReadVariableOp_12res_block_0/batch_normalization_4/ReadVariableOp_12Z
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp+res_block_0/conv2d_2/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_2/Conv2D/ReadVariableOp*res_block_0/conv2d_2/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp+res_block_0/conv2d_3/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_3/Conv2D/ReadVariableOp*res_block_0/conv2d_3/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp+res_block_0/conv2d_4/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_4/Conv2D/ReadVariableOp*res_block_0/conv2d_4/Conv2D/ReadVariableOp2?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_5/ReadVariableOp0res_block_1/batch_normalization_5/ReadVariableOp2h
2res_block_1/batch_normalization_5/ReadVariableOp_12res_block_1/batch_normalization_5/ReadVariableOp_12?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_6/ReadVariableOp0res_block_1/batch_normalization_6/ReadVariableOp2h
2res_block_1/batch_normalization_6/ReadVariableOp_12res_block_1/batch_normalization_6/ReadVariableOp_12?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_7/ReadVariableOp0res_block_1/batch_normalization_7/ReadVariableOp2h
2res_block_1/batch_normalization_7/ReadVariableOp_12res_block_1/batch_normalization_7/ReadVariableOp_12Z
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp+res_block_1/conv2d_5/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_5/Conv2D/ReadVariableOp*res_block_1/conv2d_5/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp+res_block_1/conv2d_6/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_6/Conv2D/ReadVariableOp*res_block_1/conv2d_6/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp+res_block_1/conv2d_7/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_7/Conv2D/ReadVariableOp*res_block_1/conv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?:
?
C__inference_model_layer_call_and_return_conditional_losses_43656128	
input
input_conv_43655775
input_conv_43655777
input_conv_43655779
input_conv_43655781
input_conv_43655783
input_conv_43655785
input_conv_43655787
input_conv_43655789
input_conv_43655791
input_conv_43655793
input_conv_43655795
input_conv_43655797
res_block_0_43655882
res_block_0_43655884
res_block_0_43655886
res_block_0_43655888
res_block_0_43655890
res_block_0_43655892
res_block_0_43655894
res_block_0_43655896
res_block_0_43655898
res_block_0_43655900
res_block_0_43655902
res_block_0_43655904
res_block_0_43655906
res_block_0_43655908
res_block_0_43655910
res_block_0_43655912
res_block_0_43655914
res_block_0_43655916
res_block_1_43656015
res_block_1_43656017
res_block_1_43656019
res_block_1_43656021
res_block_1_43656023
res_block_1_43656025
res_block_1_43656027
res_block_1_43656029
res_block_1_43656031
res_block_1_43656033
res_block_1_43656035
res_block_1_43656037
res_block_1_43656039
res_block_1_43656041
res_block_1_43656043
res_block_1_43656045
res_block_1_43656047
res_block_1_43656049
feature_43656100
feature_43656102
feature_43656104
feature_43656106
feature_43656108
feature_43656110
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_43655775input_conv_43655777input_conv_43655779input_conv_43655781input_conv_43655783input_conv_43655785input_conv_43655787input_conv_43655789input_conv_43655791input_conv_43655793input_conv_43655795input_conv_43655797*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536182$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_43655882res_block_0_43655884res_block_0_43655886res_block_0_43655888res_block_0_43655890res_block_0_43655892res_block_0_43655894res_block_0_43655896res_block_0_43655898res_block_0_43655900res_block_0_43655902res_block_0_43655904res_block_0_43655906res_block_0_43655908res_block_0_43655910res_block_0_43655912res_block_0_43655914res_block_0_43655916*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436544292%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_436559252
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_43656015res_block_1_43656017res_block_1_43656019res_block_1_43656021res_block_1_43656023res_block_1_43656025res_block_1_43656027res_block_1_43656029res_block_1_43656031res_block_1_43656033res_block_1_43656035res_block_1_43656037res_block_1_43656039res_block_1_43656041res_block_1_43656043res_block_1_43656045res_block_1_43656047res_block_1_43656049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436552782%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_436560582
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_43656100feature_43656102feature_43656104feature_43656106feature_43656108feature_43656110*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556622!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_436561192 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?!
?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653547
conv2d_input
conv2d_43653347
conv2d_43653349 
batch_normalization_43653414 
batch_normalization_43653416 
batch_normalization_43653418 
batch_normalization_43653420
conv2d_1_43653457
conv2d_1_43653459"
batch_normalization_1_43653524"
batch_normalization_1_43653526"
batch_normalization_1_43653528"
batch_normalization_1_43653530
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_43653347conv2d_43653349*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_436533362 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_43653414batch_normalization_43653416batch_normalization_43653418batch_normalization_43653420*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436533692-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_436534282
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_43653457conv2d_1_43653459*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_436534462"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_43653524batch_normalization_1_43653526batch_normalization_1_43653528batch_normalization_1_43653530*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436534792/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_436535382
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?!
?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653681

inputs
conv2d_43653650
conv2d_43653652 
batch_normalization_43653655 
batch_normalization_43653657 
batch_normalization_43653659 
batch_normalization_43653661
conv2d_1_43653665
conv2d_1_43653667"
batch_normalization_1_43653670"
batch_normalization_1_43653672"
batch_normalization_1_43653674"
batch_normalization_1_43653676
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_43653650conv2d_43653652*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_436533362 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_43653655batch_normalization_43653657batch_normalization_43653659batch_normalization_43653661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436533872-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_436534282
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_43653665conv2d_1_43653667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_436534462"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_43653670batch_normalization_1_43653672batch_normalization_1_43653674batch_normalization_1_43653676*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_436534972/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_436535382
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?:
?
C__inference_model_layer_call_and_return_conditional_losses_43656377

inputs
input_conv_43656257
input_conv_43656259
input_conv_43656261
input_conv_43656263
input_conv_43656265
input_conv_43656267
input_conv_43656269
input_conv_43656271
input_conv_43656273
input_conv_43656275
input_conv_43656277
input_conv_43656279
res_block_0_43656282
res_block_0_43656284
res_block_0_43656286
res_block_0_43656288
res_block_0_43656290
res_block_0_43656292
res_block_0_43656294
res_block_0_43656296
res_block_0_43656298
res_block_0_43656300
res_block_0_43656302
res_block_0_43656304
res_block_0_43656306
res_block_0_43656308
res_block_0_43656310
res_block_0_43656312
res_block_0_43656314
res_block_0_43656316
res_block_1_43656321
res_block_1_43656323
res_block_1_43656325
res_block_1_43656327
res_block_1_43656329
res_block_1_43656331
res_block_1_43656333
res_block_1_43656335
res_block_1_43656337
res_block_1_43656339
res_block_1_43656341
res_block_1_43656343
res_block_1_43656345
res_block_1_43656347
res_block_1_43656349
res_block_1_43656351
res_block_1_43656353
res_block_1_43656355
feature_43656360
feature_43656362
feature_43656364
feature_43656366
feature_43656368
feature_43656370
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_43656257input_conv_43656259input_conv_43656261input_conv_43656263input_conv_43656265input_conv_43656267input_conv_43656269input_conv_43656271input_conv_43656273input_conv_43656275input_conv_43656277input_conv_43656279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536182$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_43656282res_block_0_43656284res_block_0_43656286res_block_0_43656288res_block_0_43656290res_block_0_43656292res_block_0_43656294res_block_0_43656296res_block_0_43656298res_block_0_43656300res_block_0_43656302res_block_0_43656304res_block_0_43656306res_block_0_43656308res_block_0_43656310res_block_0_43656312res_block_0_43656314res_block_0_43656316*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436544292%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_436559252
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_43656321res_block_1_43656323res_block_1_43656325res_block_1_43656327res_block_1_43656329res_block_1_43656331res_block_1_43656333res_block_1_43656335res_block_1_43656337res_block_1_43656339res_block_1_43656341res_block_1_43656343res_block_1_43656345res_block_1_43656347res_block_1_43656349res_block_1_43656351res_block_1_43656353res_block_1_43656355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_436552782%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_436560582
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_43656360feature_43656362feature_43656364feature_43656366feature_43656368feature_43656370*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556622!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_436561192 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_4_layer_call_fn_43658892

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436542752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43653997

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_2_layer_call_fn_43658475

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_436540222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
`
D__inference_relu_1_layer_call_and_return_conditional_losses_43656058

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_43658313

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_layer_call_fn_43658231

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_436532112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_7_layer_call_fn_43659341

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436548152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43653497

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43655495

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_3_layer_call_fn_43658677

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436538662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659471

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658555

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658205

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_6_layer_call_fn_43659126

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436547152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_feature_layer_call_and_return_conditional_losses_43658082

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
activation_6/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
activation_6/Relu?
IdentityIdentityactivation_6/Relu:activations:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
??
?8
#__inference__wrapped_model_43653122	
input:
6model_input_conv_conv2d_conv2d_readvariableop_resource;
7model_input_conv_conv2d_biasadd_readvariableop_resource@
<model_input_conv_batch_normalization_readvariableop_resourceB
>model_input_conv_batch_normalization_readvariableop_1_resourceQ
Mmodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceS
Omodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource<
8model_input_conv_conv2d_1_conv2d_readvariableop_resource=
9model_input_conv_conv2d_1_biasadd_readvariableop_resourceB
>model_input_conv_batch_normalization_1_readvariableop_resourceD
@model_input_conv_batch_normalization_1_readvariableop_1_resourceS
Omodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceU
Qmodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_0_conv2d_2_conv2d_readvariableop_resource>
:model_res_block_0_conv2d_2_biasadd_readvariableop_resourceC
?model_res_block_0_batch_normalization_2_readvariableop_resourceE
Amodel_res_block_0_batch_normalization_2_readvariableop_1_resourceT
Pmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_0_conv2d_3_conv2d_readvariableop_resource>
:model_res_block_0_conv2d_3_biasadd_readvariableop_resourceC
?model_res_block_0_batch_normalization_3_readvariableop_resourceE
Amodel_res_block_0_batch_normalization_3_readvariableop_1_resourceT
Pmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_0_conv2d_4_conv2d_readvariableop_resource>
:model_res_block_0_conv2d_4_biasadd_readvariableop_resourceC
?model_res_block_0_batch_normalization_4_readvariableop_resourceE
Amodel_res_block_0_batch_normalization_4_readvariableop_1_resourceT
Pmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_1_conv2d_5_conv2d_readvariableop_resource>
:model_res_block_1_conv2d_5_biasadd_readvariableop_resourceC
?model_res_block_1_batch_normalization_5_readvariableop_resourceE
Amodel_res_block_1_batch_normalization_5_readvariableop_1_resourceT
Pmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_1_conv2d_6_conv2d_readvariableop_resource>
:model_res_block_1_conv2d_6_biasadd_readvariableop_resourceC
?model_res_block_1_batch_normalization_6_readvariableop_resourceE
Amodel_res_block_1_batch_normalization_6_readvariableop_1_resourceT
Pmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_1_conv2d_7_conv2d_readvariableop_resource>
:model_res_block_1_conv2d_7_biasadd_readvariableop_resourceC
?model_res_block_1_batch_normalization_7_readvariableop_resourceE
Amodel_res_block_1_batch_normalization_7_readvariableop_1_resourceT
Pmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource9
5model_feature_conv2d_8_conv2d_readvariableop_resource:
6model_feature_conv2d_8_biasadd_readvariableop_resource?
;model_feature_batch_normalization_8_readvariableop_resourceA
=model_feature_batch_normalization_8_readvariableop_1_resourceP
Lmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceR
Nmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?2model/feature/batch_normalization_8/ReadVariableOp?4model/feature/batch_normalization_8/ReadVariableOp_1?-model/feature/conv2d_8/BiasAdd/ReadVariableOp?,model/feature/conv2d_8/Conv2D/ReadVariableOp?Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?3model/input_conv/batch_normalization/ReadVariableOp?5model/input_conv/batch_normalization/ReadVariableOp_1?Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?5model/input_conv/batch_normalization_1/ReadVariableOp?7model/input_conv/batch_normalization_1/ReadVariableOp_1?.model/input_conv/conv2d/BiasAdd/ReadVariableOp?-model/input_conv/conv2d/Conv2D/ReadVariableOp?0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp?/model/input_conv/conv2d_1/Conv2D/ReadVariableOp?Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_0/batch_normalization_2/ReadVariableOp?8model/res_block_0/batch_normalization_2/ReadVariableOp_1?Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_0/batch_normalization_3/ReadVariableOp?8model/res_block_0/batch_normalization_3/ReadVariableOp_1?Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_0/batch_normalization_4/ReadVariableOp?8model/res_block_0/batch_normalization_4/ReadVariableOp_1?1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp?0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp?1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp?0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp?1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp?0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp?Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_1/batch_normalization_5/ReadVariableOp?8model/res_block_1/batch_normalization_5/ReadVariableOp_1?Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_1/batch_normalization_6/ReadVariableOp?8model/res_block_1/batch_normalization_6/ReadVariableOp_1?Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_1/batch_normalization_7/ReadVariableOp?8model/res_block_1/batch_normalization_7/ReadVariableOp_1?1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp?0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp?1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp?0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp?1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp?0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp?
-model/input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp6model_input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02/
-model/input_conv/conv2d/Conv2D/ReadVariableOp?
model/input_conv/conv2d/Conv2DConv2Dinput5model/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2 
model/input_conv/conv2d/Conv2D?
.model/input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp7model_input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model/input_conv/conv2d/BiasAdd/ReadVariableOp?
model/input_conv/conv2d/BiasAddBiasAdd'model/input_conv/conv2d/Conv2D:output:06model/input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2!
model/input_conv/conv2d/BiasAdd?
3model/input_conv/batch_normalization/ReadVariableOpReadVariableOp<model_input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype025
3model/input_conv/batch_normalization/ReadVariableOp?
5model/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp>model_input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5model/input_conv/batch_normalization/ReadVariableOp_1?
Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5model/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3(model/input_conv/conv2d/BiasAdd:output:0;model/input_conv/batch_normalization/ReadVariableOp:value:0=model/input_conv/batch_normalization/ReadVariableOp_1:value:0Lmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 27
5model/input_conv/batch_normalization/FusedBatchNormV3?
 model/input_conv/activation/ReluRelu9model/input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2"
 model/input_conv/activation/Relu?
/model/input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8model_input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype021
/model/input_conv/conv2d_1/Conv2D/ReadVariableOp?
 model/input_conv/conv2d_1/Conv2DConv2D.model/input_conv/activation/Relu:activations:07model/input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2"
 model/input_conv/conv2d_1/Conv2D?
0model/input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9model_input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp?
!model/input_conv/conv2d_1/BiasAddBiasAdd)model/input_conv/conv2d_1/Conv2D:output:08model/input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2#
!model/input_conv/conv2d_1/BiasAdd?
5model/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp>model_input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype027
5model/input_conv/batch_normalization_1/ReadVariableOp?
7model/input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp@model_input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7model/input_conv/batch_normalization_1/ReadVariableOp_1?
Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
7model/input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*model/input_conv/conv2d_1/BiasAdd:output:0=model/input_conv/batch_normalization_1/ReadVariableOp:value:0?model/input_conv/batch_normalization_1/ReadVariableOp_1:value:0Nmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 29
7model/input_conv/batch_normalization_1/FusedBatchNormV3?
"model/input_conv/activation_1/ReluRelu;model/input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2$
"model/input_conv/activation_1/Relu?
0model/res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp9model_res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp?
!model/res_block_0/conv2d_2/Conv2DConv2D0model/input_conv/activation_1/Relu:activations:08model/res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_0/conv2d_2/Conv2D?
1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp?
"model/res_block_0/conv2d_2/BiasAddBiasAdd*model/res_block_0/conv2d_2/Conv2D:output:09model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_0/conv2d_2/BiasAdd?
6model/res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp?model_res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_0/batch_normalization_2/ReadVariableOp?
8model/res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOpAmodel_res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_0/batch_normalization_2/ReadVariableOp_1?
Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+model/res_block_0/conv2d_2/BiasAdd:output:0>model/res_block_0/batch_normalization_2/ReadVariableOp:value:0@model/res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Omodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_0/batch_normalization_2/FusedBatchNormV3?
#model/res_block_0/activation_2/ReluRelu<model/res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_0/activation_2/Relu?
0model/res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9model_res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp?
!model/res_block_0/conv2d_3/Conv2DConv2D1model/res_block_0/activation_2/Relu:activations:08model/res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_0/conv2d_3/Conv2D?
1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp?
"model/res_block_0/conv2d_3/BiasAddBiasAdd*model/res_block_0/conv2d_3/Conv2D:output:09model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_0/conv2d_3/BiasAdd?
6model/res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp?model_res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_0/batch_normalization_3/ReadVariableOp?
8model/res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodel_res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_0/batch_normalization_3/ReadVariableOp_1?
Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+model/res_block_0/conv2d_3/BiasAdd:output:0>model/res_block_0/batch_normalization_3/ReadVariableOp:value:0@model/res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Omodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_0/batch_normalization_3/FusedBatchNormV3?
#model/res_block_0/activation_3/ReluRelu<model/res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_0/activation_3/Relu?
0model/res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp9model_res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp?
!model/res_block_0/conv2d_4/Conv2DConv2D1model/res_block_0/activation_3/Relu:activations:08model/res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_0/conv2d_4/Conv2D?
1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp?
"model/res_block_0/conv2d_4/BiasAddBiasAdd*model/res_block_0/conv2d_4/Conv2D:output:09model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_0/conv2d_4/BiasAdd?
6model/res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp?model_res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_0/batch_normalization_4/ReadVariableOp?
8model/res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOpAmodel_res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_0/batch_normalization_4/ReadVariableOp_1?
Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+model/res_block_0/conv2d_4/BiasAdd:output:0>model/res_block_0/batch_normalization_4/ReadVariableOp:value:0@model/res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Omodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_0/batch_normalization_4/FusedBatchNormV3?
 model/tf.__operators__.add/AddV2AddV20model/input_conv/activation_1/Relu:activations:0<model/res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2"
 model/tf.__operators__.add/AddV2?
model/relu_0/ReluRelu$model/tf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
model/relu_0/Relu?
0model/res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp9model_res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp?
!model/res_block_1/conv2d_5/Conv2DConv2Dmodel/relu_0/Relu:activations:08model/res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_1/conv2d_5/Conv2D?
1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp?
"model/res_block_1/conv2d_5/BiasAddBiasAdd*model/res_block_1/conv2d_5/Conv2D:output:09model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_1/conv2d_5/BiasAdd?
6model/res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp?model_res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_1/batch_normalization_5/ReadVariableOp?
8model/res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOpAmodel_res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_1/batch_normalization_5/ReadVariableOp_1?
Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+model/res_block_1/conv2d_5/BiasAdd:output:0>model/res_block_1/batch_normalization_5/ReadVariableOp:value:0@model/res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Omodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_1/batch_normalization_5/FusedBatchNormV3?
#model/res_block_1/activation_4/ReluRelu<model/res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_1/activation_4/Relu?
0model/res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp9model_res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp?
!model/res_block_1/conv2d_6/Conv2DConv2D1model/res_block_1/activation_4/Relu:activations:08model/res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_1/conv2d_6/Conv2D?
1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp?
"model/res_block_1/conv2d_6/BiasAddBiasAdd*model/res_block_1/conv2d_6/Conv2D:output:09model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_1/conv2d_6/BiasAdd?
6model/res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp?model_res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_1/batch_normalization_6/ReadVariableOp?
8model/res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpAmodel_res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_1/batch_normalization_6/ReadVariableOp_1?
Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3+model/res_block_1/conv2d_6/BiasAdd:output:0>model/res_block_1/batch_normalization_6/ReadVariableOp:value:0@model/res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Omodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_1/batch_normalization_6/FusedBatchNormV3?
#model/res_block_1/activation_5/ReluRelu<model/res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_1/activation_5/Relu?
0model/res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp9model_res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp?
!model/res_block_1/conv2d_7/Conv2DConv2D1model/res_block_1/activation_5/Relu:activations:08model/res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_1/conv2d_7/Conv2D?
1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp?
"model/res_block_1/conv2d_7/BiasAddBiasAdd*model/res_block_1/conv2d_7/Conv2D:output:09model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_1/conv2d_7/BiasAdd?
6model/res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp?model_res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_1/batch_normalization_7/ReadVariableOp?
8model/res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOpAmodel_res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_1/batch_normalization_7/ReadVariableOp_1?
Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3+model/res_block_1/conv2d_7/BiasAdd:output:0>model/res_block_1/batch_normalization_7/ReadVariableOp:value:0@model/res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Omodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_1/batch_normalization_7/FusedBatchNormV3?
"model/tf.__operators__.add_1/AddV2AddV2model/relu_0/Relu:activations:0<model/res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2$
"model/tf.__operators__.add_1/AddV2?
model/relu_1/ReluRelu&model/tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
model/relu_1/Relu?
,model/feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp5model_feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,model/feature/conv2d_8/Conv2D/ReadVariableOp?
model/feature/conv2d_8/Conv2DConv2Dmodel/relu_1/Relu:activations:04model/feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
model/feature/conv2d_8/Conv2D?
-model/feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp6model_feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-model/feature/conv2d_8/BiasAdd/ReadVariableOp?
model/feature/conv2d_8/BiasAddBiasAdd&model/feature/conv2d_8/Conv2D:output:05model/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2 
model/feature/conv2d_8/BiasAdd?
2model/feature/batch_normalization_8/ReadVariableOpReadVariableOp;model_feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype024
2model/feature/batch_normalization_8/ReadVariableOp?
4model/feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp=model_feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype026
4model/feature/batch_normalization_8/ReadVariableOp_1?
Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpLmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
4model/feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3'model/feature/conv2d_8/BiasAdd:output:0:model/feature/batch_normalization_8/ReadVariableOp:value:0<model/feature/batch_normalization_8/ReadVariableOp_1:value:0Kmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Mmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 26
4model/feature/batch_normalization_8/FusedBatchNormV3?
model/feature/activation_6/ReluRelu8model/feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2!
model/feature/activation_6/Relu?
+model/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+model/tf.compat.v1.transpose/transpose/perm?
&model/tf.compat.v1.transpose/transpose	Transpose-model/feature/activation_6/Relu:activations:04model/tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2(
&model/tf.compat.v1.transpose/transpose?
IdentityIdentity*model/tf.compat.v1.transpose/transpose:y:0D^model/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpF^model/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_13^model/feature/batch_normalization_8/ReadVariableOp5^model/feature/batch_normalization_8/ReadVariableOp_1.^model/feature/conv2d_8/BiasAdd/ReadVariableOp-^model/feature/conv2d_8/Conv2D/ReadVariableOpE^model/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpG^model/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^model/input_conv/batch_normalization/ReadVariableOp6^model/input_conv/batch_normalization/ReadVariableOp_1G^model/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^model/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_16^model/input_conv/batch_normalization_1/ReadVariableOp8^model/input_conv/batch_normalization_1/ReadVariableOp_1/^model/input_conv/conv2d/BiasAdd/ReadVariableOp.^model/input_conv/conv2d/Conv2D/ReadVariableOp1^model/input_conv/conv2d_1/BiasAdd/ReadVariableOp0^model/input_conv/conv2d_1/Conv2D/ReadVariableOpH^model/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpJ^model/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17^model/res_block_0/batch_normalization_2/ReadVariableOp9^model/res_block_0/batch_normalization_2/ReadVariableOp_1H^model/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpJ^model/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17^model/res_block_0/batch_normalization_3/ReadVariableOp9^model/res_block_0/batch_normalization_3/ReadVariableOp_1H^model/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpJ^model/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17^model/res_block_0/batch_normalization_4/ReadVariableOp9^model/res_block_0/batch_normalization_4/ReadVariableOp_12^model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp1^model/res_block_0/conv2d_2/Conv2D/ReadVariableOp2^model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp1^model/res_block_0/conv2d_3/Conv2D/ReadVariableOp2^model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp1^model/res_block_0/conv2d_4/Conv2D/ReadVariableOpH^model/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpJ^model/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17^model/res_block_1/batch_normalization_5/ReadVariableOp9^model/res_block_1/batch_normalization_5/ReadVariableOp_1H^model/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpJ^model/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17^model/res_block_1/batch_normalization_6/ReadVariableOp9^model/res_block_1/batch_normalization_6/ReadVariableOp_1H^model/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpJ^model/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17^model/res_block_1/batch_normalization_7/ReadVariableOp9^model/res_block_1/batch_normalization_7/ReadVariableOp_12^model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp1^model/res_block_1/conv2d_5/Conv2D/ReadVariableOp2^model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp1^model/res_block_1/conv2d_6/Conv2D/ReadVariableOp2^model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp1^model/res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2?
Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpCmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12h
2model/feature/batch_normalization_8/ReadVariableOp2model/feature/batch_normalization_8/ReadVariableOp2l
4model/feature/batch_normalization_8/ReadVariableOp_14model/feature/batch_normalization_8/ReadVariableOp_12^
-model/feature/conv2d_8/BiasAdd/ReadVariableOp-model/feature/conv2d_8/BiasAdd/ReadVariableOp2\
,model/feature/conv2d_8/Conv2D/ReadVariableOp,model/feature/conv2d_8/Conv2D/ReadVariableOp2?
Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3model/input_conv/batch_normalization/ReadVariableOp3model/input_conv/batch_normalization/ReadVariableOp2n
5model/input_conv/batch_normalization/ReadVariableOp_15model/input_conv/batch_normalization/ReadVariableOp_12?
Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5model/input_conv/batch_normalization_1/ReadVariableOp5model/input_conv/batch_normalization_1/ReadVariableOp2r
7model/input_conv/batch_normalization_1/ReadVariableOp_17model/input_conv/batch_normalization_1/ReadVariableOp_12`
.model/input_conv/conv2d/BiasAdd/ReadVariableOp.model/input_conv/conv2d/BiasAdd/ReadVariableOp2^
-model/input_conv/conv2d/Conv2D/ReadVariableOp-model/input_conv/conv2d/Conv2D/ReadVariableOp2d
0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp2b
/model/input_conv/conv2d_1/Conv2D/ReadVariableOp/model/input_conv/conv2d_1/Conv2D/ReadVariableOp2?
Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpGmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_0/batch_normalization_2/ReadVariableOp6model/res_block_0/batch_normalization_2/ReadVariableOp2t
8model/res_block_0/batch_normalization_2/ReadVariableOp_18model/res_block_0/batch_normalization_2/ReadVariableOp_12?
Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_0/batch_normalization_3/ReadVariableOp6model/res_block_0/batch_normalization_3/ReadVariableOp2t
8model/res_block_0/batch_normalization_3/ReadVariableOp_18model/res_block_0/batch_normalization_3/ReadVariableOp_12?
Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpGmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_0/batch_normalization_4/ReadVariableOp6model/res_block_0/batch_normalization_4/ReadVariableOp2t
8model/res_block_0/batch_normalization_4/ReadVariableOp_18model/res_block_0/batch_normalization_4/ReadVariableOp_12f
1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp2d
0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp2f
1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp2d
0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp2f
1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp2d
0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp2?
Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpGmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_1/batch_normalization_5/ReadVariableOp6model/res_block_1/batch_normalization_5/ReadVariableOp2t
8model/res_block_1/batch_normalization_5/ReadVariableOp_18model/res_block_1/batch_normalization_5/ReadVariableOp_12?
Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpGmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_1/batch_normalization_6/ReadVariableOp6model/res_block_1/batch_normalization_6/ReadVariableOp2t
8model/res_block_1/batch_normalization_6/ReadVariableOp_18model/res_block_1/batch_normalization_6/ReadVariableOp_12?
Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpGmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_1/batch_normalization_7/ReadVariableOp6model/res_block_1/batch_normalization_7/ReadVariableOp2t
8model/res_block_1/batch_normalization_7/ReadVariableOp_18model/res_block_1/batch_normalization_7/ReadVariableOp_12f
1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp2d
0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp2f
1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp2d
0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp2f
1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp2d
0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
*__inference_feature_layer_call_fn_43658141

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_436556982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659113

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43654904

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?g
?
!__inference__traced_save_43659692
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
9272
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:?:?:?:?:?@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:,%(
&
_output_shapes
:@@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:,+(
&
_output_shapes
:@@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
::7

_output_shapes
: 
?
f
J__inference_activation_1_layer_call_and_return_conditional_losses_43653538

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_43654871

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_43657459

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_436566132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659095

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
.__inference_res_block_0_layer_call_fn_43657782

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_436544292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
f
J__inference_activation_4_layer_call_and_return_conditional_losses_43659053

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658646

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
K
/__inference_activation_2_layer_call_fn_43658609

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_436541142
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?.
?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654378
conv2d_2_input
conv2d_2_43654333
conv2d_2_43654335"
batch_normalization_2_43654338"
batch_normalization_2_43654340"
batch_normalization_2_43654342"
batch_normalization_2_43654344
conv2d_3_43654348
conv2d_3_43654350"
batch_normalization_3_43654353"
batch_normalization_3_43654355"
batch_normalization_3_43654357"
batch_normalization_3_43654359
conv2d_4_43654363
conv2d_4_43654365"
batch_normalization_4_43654368"
batch_normalization_4_43654370"
batch_normalization_4_43654372"
batch_normalization_4_43654374
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_43654333conv2d_2_43654335*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_436540222"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_43654338batch_normalization_2_43654340batch_normalization_2_43654342batch_normalization_2_43654344*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436540732/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_436541142
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_43654348conv2d_3_43654350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_436541322"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_43654353batch_normalization_3_43654355batch_normalization_3_43654357batch_normalization_3_43654359*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436541832/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_436542242
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_43654363conv2d_4_43654365*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_436542422"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_43654368batch_normalization_4_43654370batch_normalization_4_43654372batch_normalization_4_43654374*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436542932/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
f
J__inference_activation_2_layer_call_and_return_conditional_losses_43654114

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?.
?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654518

inputs
conv2d_2_43654473
conv2d_2_43654475"
batch_normalization_2_43654478"
batch_normalization_2_43654480"
batch_normalization_2_43654482"
batch_normalization_2_43654484
conv2d_3_43654488
conv2d_3_43654490"
batch_normalization_3_43654493"
batch_normalization_3_43654495"
batch_normalization_3_43654497"
batch_normalization_3_43654499
conv2d_4_43654503
conv2d_4_43654505"
batch_normalization_4_43654508"
batch_normalization_4_43654510"
batch_normalization_4_43654512"
batch_normalization_4_43654514
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_43654473conv2d_2_43654475*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_436540222"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_43654478batch_normalization_2_43654480batch_normalization_2_43654482batch_normalization_2_43654484*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436540732/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_436541142
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_43654488conv2d_3_43654490*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_436541322"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_43654493batch_normalization_3_43654495batch_normalization_3_43654497batch_normalization_3_43654499*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436541832/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_436542242
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_43654503conv2d_4_43654505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_436542422"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_43654508batch_normalization_4_43654510batch_normalization_4_43654512batch_normalization_4_43654514*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436542932/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
E__inference_feature_layer_call_and_return_conditional_losses_43655662

inputs
conv2d_8_43655646
conv2d_8_43655648"
batch_normalization_8_43655651"
batch_normalization_8_43655653"
batch_normalization_8_43655655"
batch_normalization_8_43655657
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_43655646conv2d_8_43655648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_436555202"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_43655651batch_normalization_8_43655653batch_normalization_8_43655655batch_normalization_8_43655657*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_436555532/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_436556122
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
-__inference_input_conv_layer_call_fn_43657609

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_436536812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658511

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?.
?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654330
conv2d_2_input
conv2d_2_43654033
conv2d_2_43654035"
batch_normalization_2_43654100"
batch_normalization_2_43654102"
batch_normalization_2_43654104"
batch_normalization_2_43654106
conv2d_3_43654143
conv2d_3_43654145"
batch_normalization_3_43654210"
batch_normalization_3_43654212"
batch_normalization_3_43654214"
batch_normalization_3_43654216
conv2d_4_43654253
conv2d_4_43654255"
batch_normalization_4_43654320"
batch_normalization_4_43654322"
batch_normalization_4_43654324"
batch_normalization_4_43654326
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_43654033conv2d_2_43654035*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_436540222"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_43654100batch_normalization_2_43654102batch_normalization_2_43654104batch_normalization_2_43654106*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436540552/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_436541142
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_43654143conv2d_3_43654145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_436541322"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_43654210batch_normalization_3_43654212batch_normalization_3_43654214batch_normalization_3_43654216*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_436541652/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_436542242
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_43654253conv2d_4_43654255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_436542422"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_43654320batch_normalization_4_43654322batch_normalization_4_43654324batch_normalization_4_43654326*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_436542752/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658187

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?.
?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655278

inputs
conv2d_5_43655233
conv2d_5_43655235"
batch_normalization_5_43655238"
batch_normalization_5_43655240"
batch_normalization_5_43655242"
batch_normalization_5_43655244
conv2d_6_43655248
conv2d_6_43655250"
batch_normalization_6_43655253"
batch_normalization_6_43655255"
batch_normalization_6_43655257"
batch_normalization_6_43655259
conv2d_7_43655263
conv2d_7_43655265"
batch_normalization_7_43655268"
batch_normalization_7_43655270"
batch_normalization_7_43655272"
batch_normalization_7_43655274
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_43655233conv2d_5_43655235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_436548712"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_43655238batch_normalization_5_43655240batch_normalization_5_43655242batch_normalization_5_43655244*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436549042/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_436549632
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_43655248conv2d_6_43655250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_436549812"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_43655253batch_normalization_6_43655255batch_normalization_6_43655257batch_normalization_6_43655259*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_436550142/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_436550732
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_43655263conv2d_7_43655265*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_436550912"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_43655268batch_normalization_7_43655270batch_normalization_7_43655272batch_normalization_7_43655274*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_436551242/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_2_layer_call_fn_43658524

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_436540552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_43658772

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
f
J__inference_activation_3_layer_call_and_return_conditional_losses_43654224

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_5_layer_call_fn_43659035

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_436549042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
f
J__inference_activation_5_layer_call_and_return_conditional_losses_43659206

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658249

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659248

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
E
)__inference_relu_0_layer_call_fn_43657833

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_436559252
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input7
serving_default_input:0??????????J
feature_linear8
StatefulPartitionedCall:0?????????Rtensorflow/serving/predict:˫

??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10

signatures
#_self_saveable_object_factories
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "input_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_0", "inbound_nodes": [[["input_conv", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["input_conv", 1, 0, {"y": ["res_block_0", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_0", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_1", "inbound_nodes": [[["relu_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["relu_0", 0, 0, {"y": ["res_block_1", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "feature", "inbound_nodes": [[["relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["feature", 1, 0, {"perm": {"class_name": "__tuple__", "items": [0, 1, 3, 2]}}]]}, {"class_name": "Activation", "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}, "name": "feature_linear", "inbound_nodes": [[["tf.compat.v1.transpose", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["feature_linear", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "input_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_0", "inbound_nodes": [[["input_conv", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["input_conv", 1, 0, {"y": ["res_block_0", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_0", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_1", "inbound_nodes": [[["relu_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["relu_0", 0, 0, {"y": ["res_block_1", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "feature", "inbound_nodes": [[["relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["feature", 1, 0, {"perm": {"class_name": "__tuple__", "items": [0, 1, 3, 2]}}]]}, {"class_name": "Activation", "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}, "name": "feature_linear", "inbound_nodes": [[["tf.compat.v1.transpose", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["feature_linear", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?2
_tf_keras_sequential?2{"class_name": "Sequential", "name": "input_conv", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}}}
?I
 layer_with_weights-0
 layer-0
!layer_with_weights-1
!layer-1
"layer-2
#layer_with_weights-2
#layer-3
$layer_with_weights-3
$layer-4
%layer-5
&layer_with_weights-4
&layer-6
'layer_with_weights-5
'layer-7
#(_self_saveable_object_factories
)	variables
*trainable_variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?F
_tf_keras_sequential?F{"class_name": "Sequential", "name": "res_block_0", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}
?
#-_self_saveable_object_factories
.	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}}
?
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "relu_0", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}}
?I
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6layer-2
7layer_with_weights-2
7layer-3
8layer_with_weights-3
8layer-4
9layer-5
:layer_with_weights-4
:layer-6
;layer_with_weights-5
;layer-7
#<_self_saveable_object_factories
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?F
_tf_keras_sequential?F{"class_name": "Sequential", "name": "res_block_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}
?
#A_self_saveable_object_factories
B	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}}
?
#C_self_saveable_object_factories
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "relu_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}}
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
#K_self_saveable_object_factories
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "feature", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}}}
?
#P_self_saveable_object_factories
Q	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.transpose", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}}
?
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "feature_linear", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}}
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
"
	optimizer
 "
trackable_dict_wrapper
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
?


Wkernel
Xbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}}
?	
	?axis
	Ygamma
Zbeta
[moving_mean
\moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


]kernel
^bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
?	
	?axis
	_gamma
`beta
amoving_mean
bmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}
 "
trackable_dict_wrapper
v
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


ckernel
dbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


ikernel
jbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	kgamma
lbeta
mmoving_mean
nmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


okernel
pbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	qgamma
rbeta
smoving_mean
tmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
 "
trackable_dict_wrapper
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
)	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
*trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
0	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
1trainable_variables
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


ukernel
vbias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	wgamma
xbeta
ymoving_mean
zmoving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


{kernel
|bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	}gamma
~beta
moving_mean
?moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_7", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
 "
trackable_dict_wrapper
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
=	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
>trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
D	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
Etrainable_variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_8", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 8]}}
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}
 "
trackable_dict_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
L	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
Mtrainable_variables
Nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
S	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
Ttrainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&?2conv2d/kernel
:?2conv2d/bias
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
*:(?@2conv2d_1/kernel
:@2conv2d_1/bias
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
):'@@2conv2d_6/kernel
:@2conv2d_6/bias
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
):'@2conv2d_8/kernel
:2conv2d_8/bias
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
_0
`1
a2
b3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
v
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17"
trackable_list_wrapper
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
=
}0
~1
2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17"
trackable_list_wrapper
X
40
51
62
73
84
95
:6
;7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
_0
`1
a2
b3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
=
}0
~1
2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
C__inference_model_layer_call_and_return_conditional_losses_43656251
C__inference_model_layer_call_and_return_conditional_losses_43657036
C__inference_model_layer_call_and_return_conditional_losses_43657233
C__inference_model_layer_call_and_return_conditional_losses_43656128?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_43653122?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
input??????????
?2?
(__inference_model_layer_call_fn_43657459
(__inference_model_layer_call_fn_43656724
(__inference_model_layer_call_fn_43656488
(__inference_model_layer_call_fn_43657346?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653581
H__inference_input_conv_layer_call_and_return_conditional_losses_43657505
H__inference_input_conv_layer_call_and_return_conditional_losses_43657551
H__inference_input_conv_layer_call_and_return_conditional_losses_43653547?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_input_conv_layer_call_fn_43657580
-__inference_input_conv_layer_call_fn_43653645
-__inference_input_conv_layer_call_fn_43657609
-__inference_input_conv_layer_call_fn_43653708?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654378
I__inference_res_block_0_layer_call_and_return_conditional_losses_43657675
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654330
I__inference_res_block_0_layer_call_and_return_conditional_losses_43657741?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_res_block_0_layer_call_fn_43657823
.__inference_res_block_0_layer_call_fn_43657782
.__inference_res_block_0_layer_call_fn_43654468
.__inference_res_block_0_layer_call_fn_43654557?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_relu_0_layer_call_and_return_conditional_losses_43657828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_relu_0_layer_call_fn_43657833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43657965
I__inference_res_block_1_layer_call_and_return_conditional_losses_43657899
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655227
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655179?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_res_block_1_layer_call_fn_43655406
.__inference_res_block_1_layer_call_fn_43658006
.__inference_res_block_1_layer_call_fn_43658047
.__inference_res_block_1_layer_call_fn_43655317?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_relu_1_layer_call_and_return_conditional_losses_43658052?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_relu_1_layer_call_fn_43658057?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_feature_layer_call_and_return_conditional_losses_43655621
E__inference_feature_layer_call_and_return_conditional_losses_43658082
E__inference_feature_layer_call_and_return_conditional_losses_43658107
E__inference_feature_layer_call_and_return_conditional_losses_43655640?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_feature_layer_call_fn_43658124
*__inference_feature_layer_call_fn_43658141
*__inference_feature_layer_call_fn_43655713
*__inference_feature_layer_call_fn_43655677?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_feature_linear_layer_call_and_return_conditional_losses_43658145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_feature_linear_layer_call_fn_43658150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_43656839input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_layer_call_and_return_conditional_losses_43658160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_layer_call_fn_43658169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658249
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658205
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658187
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658267?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_layer_call_fn_43658218
6__inference_batch_normalization_layer_call_fn_43658293
6__inference_batch_normalization_layer_call_fn_43658231
6__inference_batch_normalization_layer_call_fn_43658280?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_activation_layer_call_and_return_conditional_losses_43658298?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_layer_call_fn_43658303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_43658313?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_1_layer_call_fn_43658322?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658402
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658358
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658340
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658420?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_1_layer_call_fn_43658433
8__inference_batch_normalization_1_layer_call_fn_43658446
8__inference_batch_normalization_1_layer_call_fn_43658384
8__inference_batch_normalization_1_layer_call_fn_43658371?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_activation_1_layer_call_and_return_conditional_losses_43658451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_1_layer_call_fn_43658456?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_43658466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_2_layer_call_fn_43658475?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658573
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658555
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658511
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658493?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_2_layer_call_fn_43658599
8__inference_batch_normalization_2_layer_call_fn_43658524
8__inference_batch_normalization_2_layer_call_fn_43658586
8__inference_batch_normalization_2_layer_call_fn_43658537?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_activation_2_layer_call_and_return_conditional_losses_43658604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_2_layer_call_fn_43658609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_43658619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_3_layer_call_fn_43658628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658664
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658708
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658646
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658726?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_3_layer_call_fn_43658677
8__inference_batch_normalization_3_layer_call_fn_43658739
8__inference_batch_normalization_3_layer_call_fn_43658690
8__inference_batch_normalization_3_layer_call_fn_43658752?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_activation_3_layer_call_and_return_conditional_losses_43658757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_3_layer_call_fn_43658762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_43658772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_4_layer_call_fn_43658781?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658861
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658817
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658799
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658879?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_4_layer_call_fn_43658843
8__inference_batch_normalization_4_layer_call_fn_43658892
8__inference_batch_normalization_4_layer_call_fn_43658830
8__inference_batch_normalization_4_layer_call_fn_43658905?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_43658915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_5_layer_call_fn_43658924?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43658960
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43658942
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43659022
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43659004?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_5_layer_call_fn_43659035
8__inference_batch_normalization_5_layer_call_fn_43658973
8__inference_batch_normalization_5_layer_call_fn_43659048
8__inference_batch_normalization_5_layer_call_fn_43658986?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_activation_4_layer_call_and_return_conditional_losses_43659053?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_4_layer_call_fn_43659058?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_43659068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_6_layer_call_fn_43659077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659095
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659113
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659157
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659175?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_6_layer_call_fn_43659126
8__inference_batch_normalization_6_layer_call_fn_43659139
8__inference_batch_normalization_6_layer_call_fn_43659188
8__inference_batch_normalization_6_layer_call_fn_43659201?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_activation_5_layer_call_and_return_conditional_losses_43659206?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_5_layer_call_fn_43659211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_43659221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_7_layer_call_fn_43659230?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659310
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659248
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659266
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659328?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_7_layer_call_fn_43659341
8__inference_batch_normalization_7_layer_call_fn_43659292
8__inference_batch_normalization_7_layer_call_fn_43659354
8__inference_batch_normalization_7_layer_call_fn_43659279?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_43659364?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_8_layer_call_fn_43659373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659391
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659409
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659453
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659471?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_8_layer_call_fn_43659497
8__inference_batch_normalization_8_layer_call_fn_43659484
8__inference_batch_normalization_8_layer_call_fn_43659435
8__inference_batch_normalization_8_layer_call_fn_43659422?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_activation_6_layer_call_and_return_conditional_losses_43659502?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_6_layer_call_fn_43659507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_43653122?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????7?4
-?*
(?%
input??????????
? "G?D
B
feature_linear0?-
feature_linear?????????R?
J__inference_activation_1_layer_call_and_return_conditional_losses_43658451h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
/__inference_activation_1_layer_call_fn_43658456[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
J__inference_activation_2_layer_call_and_return_conditional_losses_43658604h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
/__inference_activation_2_layer_call_fn_43658609[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
J__inference_activation_3_layer_call_and_return_conditional_losses_43658757h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
/__inference_activation_3_layer_call_fn_43658762[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
J__inference_activation_4_layer_call_and_return_conditional_losses_43659053h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
/__inference_activation_4_layer_call_fn_43659058[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
J__inference_activation_5_layer_call_and_return_conditional_losses_43659206h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
/__inference_activation_5_layer_call_fn_43659211[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
J__inference_activation_6_layer_call_and_return_conditional_losses_43659502h7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????R
? ?
/__inference_activation_6_layer_call_fn_43659507[7?4
-?*
(?%
inputs?????????R
? " ??????????R?
H__inference_activation_layer_call_and_return_conditional_losses_43658298l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
-__inference_activation_layer_call_fn_43658303_9?6
/?,
*?'
inputs???????????
? ""?????????????
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658340?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658358?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658402r_`ab;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43658420r_`ab;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
8__inference_batch_normalization_1_layer_call_fn_43658371?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_1_layer_call_fn_43658384?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_1_layer_call_fn_43658433e_`ab;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_1_layer_call_fn_43658446e_`ab;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658493refgh;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658511refgh;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658555?efghM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43658573?efghM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_2_layer_call_fn_43658524eefgh;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_2_layer_call_fn_43658537eefgh;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
8__inference_batch_normalization_2_layer_call_fn_43658586?efghM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_2_layer_call_fn_43658599?efghM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658646?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658664?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658708rklmn;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_43658726rklmn;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
8__inference_batch_normalization_3_layer_call_fn_43658677?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_3_layer_call_fn_43658690?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_3_layer_call_fn_43658739eklmn;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_3_layer_call_fn_43658752eklmn;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658799?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658817?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658861rqrst;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43658879rqrst;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
8__inference_batch_normalization_4_layer_call_fn_43658830?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_4_layer_call_fn_43658843?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_4_layer_call_fn_43658892eqrst;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_4_layer_call_fn_43658905eqrst;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43658942?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43658960?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43659004rwxyz;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43659022rwxyz;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
8__inference_batch_normalization_5_layer_call_fn_43658973?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_5_layer_call_fn_43658986?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_5_layer_call_fn_43659035ewxyz;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_5_layer_call_fn_43659048ewxyz;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659095?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659113?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659157s}~?;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_43659175s}~?;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
8__inference_batch_normalization_6_layer_call_fn_43659126?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_6_layer_call_fn_43659139?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_6_layer_call_fn_43659188f}~?;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_6_layer_call_fn_43659201f}~?;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659248v????;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659266v????;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659310?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_43659328?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_7_layer_call_fn_43659279i????;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
8__inference_batch_normalization_7_layer_call_fn_43659292i????;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
8__inference_batch_normalization_7_layer_call_fn_43659341?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
8__inference_batch_normalization_7_layer_call_fn_43659354?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659391v????;?8
1?.
(?%
inputs?????????R
p
? "-?*
#? 
0?????????R
? ?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659409v????;?8
1?.
(?%
inputs?????????R
p 
? "-?*
#? 
0?????????R
? ?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659453?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_43659471?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_8_layer_call_fn_43659422i????;?8
1?.
(?%
inputs?????????R
p
? " ??????????R?
8__inference_batch_normalization_8_layer_call_fn_43659435i????;?8
1?.
(?%
inputs?????????R
p 
? " ??????????R?
8__inference_batch_normalization_8_layer_call_fn_43659484?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_8_layer_call_fn_43659497?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658187?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658205?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658249vYZ[\=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_43658267vYZ[\=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
6__inference_batch_normalization_layer_call_fn_43658218?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_layer_call_fn_43658231?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_layer_call_fn_43658280iYZ[\=?:
3?0
*?'
inputs???????????
p
? ""?????????????
6__inference_batch_normalization_layer_call_fn_43658293iYZ[\=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
F__inference_conv2d_1_layer_call_and_return_conditional_losses_43658313n]^9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_1_layer_call_fn_43658322a]^9?6
/?,
*?'
inputs???????????
? " ??????????R@?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_43658466lcd7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_2_layer_call_fn_43658475_cd7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_43658619lij7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_3_layer_call_fn_43658628_ij7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_43658772lop7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_4_layer_call_fn_43658781_op7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_43658915luv7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_5_layer_call_fn_43658924_uv7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_43659068l{|7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_6_layer_call_fn_43659077_{|7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_43659221n??7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
+__inference_conv2d_7_layer_call_fn_43659230a??7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_43659364n??7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R
? ?
+__inference_conv2d_8_layer_call_fn_43659373a??7?4
-?*
(?%
inputs?????????R@
? " ??????????R?
D__inference_conv2d_layer_call_and_return_conditional_losses_43658160oWX8?5
.?+
)?&
inputs??????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_layer_call_fn_43658169bWX8?5
.?+
)?&
inputs??????????
? ""?????????????
E__inference_feature_layer_call_and_return_conditional_losses_43655621???????G?D
=?:
0?-
conv2d_8_input?????????R@
p

 
? "-?*
#? 
0?????????R
? ?
E__inference_feature_layer_call_and_return_conditional_losses_43655640???????G?D
=?:
0?-
conv2d_8_input?????????R@
p 

 
? "-?*
#? 
0?????????R
? ?
E__inference_feature_layer_call_and_return_conditional_losses_43658082~????????<
5?2
(?%
inputs?????????R@
p

 
? "-?*
#? 
0?????????R
? ?
E__inference_feature_layer_call_and_return_conditional_losses_43658107~????????<
5?2
(?%
inputs?????????R@
p 

 
? "-?*
#? 
0?????????R
? ?
*__inference_feature_layer_call_fn_43655677y??????G?D
=?:
0?-
conv2d_8_input?????????R@
p

 
? " ??????????R?
*__inference_feature_layer_call_fn_43655713y??????G?D
=?:
0?-
conv2d_8_input?????????R@
p 

 
? " ??????????R?
*__inference_feature_layer_call_fn_43658124q????????<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R?
*__inference_feature_layer_call_fn_43658141q????????<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R?
L__inference_feature_linear_layer_call_and_return_conditional_losses_43658145h7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????R
? ?
1__inference_feature_linear_layer_call_fn_43658150[7?4
-?*
(?%
inputs?????????R
? " ??????????R?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653547?WXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p

 
? "-?*
#? 
0?????????R@
? ?
H__inference_input_conv_layer_call_and_return_conditional_losses_43653581?WXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p 

 
? "-?*
#? 
0?????????R@
? ?
H__inference_input_conv_layer_call_and_return_conditional_losses_43657505WXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p

 
? "-?*
#? 
0?????????R@
? ?
H__inference_input_conv_layer_call_and_return_conditional_losses_43657551WXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p 

 
? "-?*
#? 
0?????????R@
? ?
-__inference_input_conv_layer_call_fn_43653645xWXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p

 
? " ??????????R@?
-__inference_input_conv_layer_call_fn_43653708xWXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p 

 
? " ??????????R@?
-__inference_input_conv_layer_call_fn_43657580rWXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p

 
? " ??????????R@?
-__inference_input_conv_layer_call_fn_43657609rWXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????R@?
C__inference_model_layer_call_and_return_conditional_losses_43656128?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p

 
? "-?*
#? 
0?????????R
? ?
C__inference_model_layer_call_and_return_conditional_losses_43656251?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p 

 
? "-?*
#? 
0?????????R
? ?
C__inference_model_layer_call_and_return_conditional_losses_43657036?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p

 
? "-?*
#? 
0?????????R
? ?
C__inference_model_layer_call_and_return_conditional_losses_43657233?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p 

 
? "-?*
#? 
0?????????R
? ?
(__inference_model_layer_call_fn_43656488?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p

 
? " ??????????R?
(__inference_model_layer_call_fn_43656724?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p 

 
? " ??????????R?
(__inference_model_layer_call_fn_43657346?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p

 
? " ??????????R?
(__inference_model_layer_call_fn_43657459?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????R?
D__inference_relu_0_layer_call_and_return_conditional_losses_43657828h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
)__inference_relu_0_layer_call_fn_43657833[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
D__inference_relu_1_layer_call_and_return_conditional_losses_43658052h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
)__inference_relu_1_layer_call_fn_43658057[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654330?cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43654378?cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43657675?cdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
I__inference_res_block_0_layer_call_and_return_conditional_losses_43657741?cdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
.__inference_res_block_0_layer_call_fn_43654468cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p

 
? " ??????????R@?
.__inference_res_block_0_layer_call_fn_43654557cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p 

 
? " ??????????R@?
.__inference_res_block_0_layer_call_fn_43657782wcdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R@?
.__inference_res_block_0_layer_call_fn_43657823wcdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R@?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655179?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43655227?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43657899?uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
I__inference_res_block_1_layer_call_and_return_conditional_losses_43657965?uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
.__inference_res_block_1_layer_call_fn_43655317?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p

 
? " ??????????R@?
.__inference_res_block_1_layer_call_fn_43655406?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p 

 
? " ??????????R@?
.__inference_res_block_1_layer_call_fn_43658006~uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R@?
.__inference_res_block_1_layer_call_fn_43658047~uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R@?
&__inference_signature_wrapper_43656839?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
? 
6?3
1
input(?%
input??????????"G?D
B
feature_linear0?-
feature_linear?????????R