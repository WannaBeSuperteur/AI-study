Ơ#
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.32v2.9.2-107-ga5ed5f39b678�
�
8Adam/regression__other__person__model/dense_final/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/regression__other__person__model/dense_final/bias/v
�
LAdam/regression__other__person__model/dense_final/bias/v/Read/ReadVariableOpReadVariableOp8Adam/regression__other__person__model/dense_final/bias/v*
_output_shapes
:*
dtype0
�
:Adam/regression__other__person__model/dense_final/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:B*K
shared_name<:Adam/regression__other__person__model/dense_final/kernel/v
�
NAdam/regression__other__person__model/dense_final/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/regression__other__person__model/dense_final/kernel/v*
_output_shapes

:B*
dtype0
�
4Adam/regression__other__person__model/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64Adam/regression__other__person__model/dense_1/bias/v
�
HAdam/regression__other__person__model/dense_1/bias/v/Read/ReadVariableOpReadVariableOp4Adam/regression__other__person__model/dense_1/bias/v*
_output_shapes
:@*
dtype0
�
6Adam/regression__other__person__model/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*G
shared_name86Adam/regression__other__person__model/dense_1/kernel/v
�
JAdam/regression__other__person__model/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/regression__other__person__model/dense_1/kernel/v*
_output_shapes
:	�@*
dtype0
�
4Adam/regression__other__person__model/dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64Adam/regression__other__person__model/dense_0/bias/v
�
HAdam/regression__other__person__model/dense_0/bias/v/Read/ReadVariableOpReadVariableOp4Adam/regression__other__person__model/dense_0/bias/v*
_output_shapes	
:�*
dtype0
�
6Adam/regression__other__person__model/dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�4�*G
shared_name86Adam/regression__other__person__model/dense_0/kernel/v
�
JAdam/regression__other__person__model/dense_0/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/regression__other__person__model/dense_0/kernel/v* 
_output_shapes
:
�4�*
dtype0
�
3Adam/regression__other__person__model/conv3R/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/regression__other__person__model/conv3R/bias/v
�
GAdam/regression__other__person__model/conv3R/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv3R/bias/v*
_output_shapes	
:�*
dtype0
�
5Adam/regression__other__person__model/conv3R/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*F
shared_name75Adam/regression__other__person__model/conv3R/kernel/v
�
IAdam/regression__other__person__model/conv3R/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv3R/kernel/v*'
_output_shapes
:@�*
dtype0
�
3Adam/regression__other__person__model/conv2R/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/regression__other__person__model/conv2R/bias/v
�
GAdam/regression__other__person__model/conv2R/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv2R/bias/v*
_output_shapes
:@*
dtype0
�
5Adam/regression__other__person__model/conv2R/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*F
shared_name75Adam/regression__other__person__model/conv2R/kernel/v
�
IAdam/regression__other__person__model/conv2R/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv2R/kernel/v*&
_output_shapes
: @*
dtype0
�
3Adam/regression__other__person__model/conv1R/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv1R/bias/v
�
GAdam/regression__other__person__model/conv1R/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv1R/bias/v*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv1R/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75Adam/regression__other__person__model/conv1R/kernel/v
�
IAdam/regression__other__person__model/conv1R/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv1R/kernel/v*&
_output_shapes
:  *
dtype0
�
3Adam/regression__other__person__model/conv0R/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv0R/bias/v
�
GAdam/regression__other__person__model/conv0R/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv0R/bias/v*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv0R/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/regression__other__person__model/conv0R/kernel/v
�
IAdam/regression__other__person__model/conv0R/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv0R/kernel/v*&
_output_shapes
: *
dtype0
�
3Adam/regression__other__person__model/conv3L/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/regression__other__person__model/conv3L/bias/v
�
GAdam/regression__other__person__model/conv3L/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv3L/bias/v*
_output_shapes	
:�*
dtype0
�
5Adam/regression__other__person__model/conv3L/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*F
shared_name75Adam/regression__other__person__model/conv3L/kernel/v
�
IAdam/regression__other__person__model/conv3L/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv3L/kernel/v*'
_output_shapes
:@�*
dtype0
�
3Adam/regression__other__person__model/conv2L/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/regression__other__person__model/conv2L/bias/v
�
GAdam/regression__other__person__model/conv2L/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv2L/bias/v*
_output_shapes
:@*
dtype0
�
5Adam/regression__other__person__model/conv2L/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*F
shared_name75Adam/regression__other__person__model/conv2L/kernel/v
�
IAdam/regression__other__person__model/conv2L/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv2L/kernel/v*&
_output_shapes
: @*
dtype0
�
3Adam/regression__other__person__model/conv1L/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv1L/bias/v
�
GAdam/regression__other__person__model/conv1L/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv1L/bias/v*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv1L/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75Adam/regression__other__person__model/conv1L/kernel/v
�
IAdam/regression__other__person__model/conv1L/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv1L/kernel/v*&
_output_shapes
:  *
dtype0
�
3Adam/regression__other__person__model/conv0L/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv0L/bias/v
�
GAdam/regression__other__person__model/conv0L/bias/v/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv0L/bias/v*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv0L/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/regression__other__person__model/conv0L/kernel/v
�
IAdam/regression__other__person__model/conv0L/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv0L/kernel/v*&
_output_shapes
: *
dtype0
�
8Adam/regression__other__person__model/dense_final/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/regression__other__person__model/dense_final/bias/m
�
LAdam/regression__other__person__model/dense_final/bias/m/Read/ReadVariableOpReadVariableOp8Adam/regression__other__person__model/dense_final/bias/m*
_output_shapes
:*
dtype0
�
:Adam/regression__other__person__model/dense_final/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:B*K
shared_name<:Adam/regression__other__person__model/dense_final/kernel/m
�
NAdam/regression__other__person__model/dense_final/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/regression__other__person__model/dense_final/kernel/m*
_output_shapes

:B*
dtype0
�
4Adam/regression__other__person__model/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64Adam/regression__other__person__model/dense_1/bias/m
�
HAdam/regression__other__person__model/dense_1/bias/m/Read/ReadVariableOpReadVariableOp4Adam/regression__other__person__model/dense_1/bias/m*
_output_shapes
:@*
dtype0
�
6Adam/regression__other__person__model/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*G
shared_name86Adam/regression__other__person__model/dense_1/kernel/m
�
JAdam/regression__other__person__model/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/regression__other__person__model/dense_1/kernel/m*
_output_shapes
:	�@*
dtype0
�
4Adam/regression__other__person__model/dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64Adam/regression__other__person__model/dense_0/bias/m
�
HAdam/regression__other__person__model/dense_0/bias/m/Read/ReadVariableOpReadVariableOp4Adam/regression__other__person__model/dense_0/bias/m*
_output_shapes	
:�*
dtype0
�
6Adam/regression__other__person__model/dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�4�*G
shared_name86Adam/regression__other__person__model/dense_0/kernel/m
�
JAdam/regression__other__person__model/dense_0/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/regression__other__person__model/dense_0/kernel/m* 
_output_shapes
:
�4�*
dtype0
�
3Adam/regression__other__person__model/conv3R/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/regression__other__person__model/conv3R/bias/m
�
GAdam/regression__other__person__model/conv3R/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv3R/bias/m*
_output_shapes	
:�*
dtype0
�
5Adam/regression__other__person__model/conv3R/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*F
shared_name75Adam/regression__other__person__model/conv3R/kernel/m
�
IAdam/regression__other__person__model/conv3R/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv3R/kernel/m*'
_output_shapes
:@�*
dtype0
�
3Adam/regression__other__person__model/conv2R/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/regression__other__person__model/conv2R/bias/m
�
GAdam/regression__other__person__model/conv2R/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv2R/bias/m*
_output_shapes
:@*
dtype0
�
5Adam/regression__other__person__model/conv2R/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*F
shared_name75Adam/regression__other__person__model/conv2R/kernel/m
�
IAdam/regression__other__person__model/conv2R/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv2R/kernel/m*&
_output_shapes
: @*
dtype0
�
3Adam/regression__other__person__model/conv1R/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv1R/bias/m
�
GAdam/regression__other__person__model/conv1R/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv1R/bias/m*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv1R/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75Adam/regression__other__person__model/conv1R/kernel/m
�
IAdam/regression__other__person__model/conv1R/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv1R/kernel/m*&
_output_shapes
:  *
dtype0
�
3Adam/regression__other__person__model/conv0R/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv0R/bias/m
�
GAdam/regression__other__person__model/conv0R/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv0R/bias/m*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv0R/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/regression__other__person__model/conv0R/kernel/m
�
IAdam/regression__other__person__model/conv0R/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv0R/kernel/m*&
_output_shapes
: *
dtype0
�
3Adam/regression__other__person__model/conv3L/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53Adam/regression__other__person__model/conv3L/bias/m
�
GAdam/regression__other__person__model/conv3L/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv3L/bias/m*
_output_shapes	
:�*
dtype0
�
5Adam/regression__other__person__model/conv3L/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*F
shared_name75Adam/regression__other__person__model/conv3L/kernel/m
�
IAdam/regression__other__person__model/conv3L/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv3L/kernel/m*'
_output_shapes
:@�*
dtype0
�
3Adam/regression__other__person__model/conv2L/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/regression__other__person__model/conv2L/bias/m
�
GAdam/regression__other__person__model/conv2L/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv2L/bias/m*
_output_shapes
:@*
dtype0
�
5Adam/regression__other__person__model/conv2L/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*F
shared_name75Adam/regression__other__person__model/conv2L/kernel/m
�
IAdam/regression__other__person__model/conv2L/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv2L/kernel/m*&
_output_shapes
: @*
dtype0
�
3Adam/regression__other__person__model/conv1L/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv1L/bias/m
�
GAdam/regression__other__person__model/conv1L/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv1L/bias/m*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv1L/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75Adam/regression__other__person__model/conv1L/kernel/m
�
IAdam/regression__other__person__model/conv1L/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv1L/kernel/m*&
_output_shapes
:  *
dtype0
�
3Adam/regression__other__person__model/conv0L/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/regression__other__person__model/conv0L/bias/m
�
GAdam/regression__other__person__model/conv0L/bias/m/Read/ReadVariableOpReadVariableOp3Adam/regression__other__person__model/conv0L/bias/m*
_output_shapes
: *
dtype0
�
5Adam/regression__other__person__model/conv0L/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/regression__other__person__model/conv0L/kernel/m
�
IAdam/regression__other__person__model/conv0L/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/regression__other__person__model/conv0L/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
1regression__other__person__model/dense_final/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31regression__other__person__model/dense_final/bias
�
Eregression__other__person__model/dense_final/bias/Read/ReadVariableOpReadVariableOp1regression__other__person__model/dense_final/bias*
_output_shapes
:*
dtype0
�
3regression__other__person__model/dense_final/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:B*D
shared_name53regression__other__person__model/dense_final/kernel
�
Gregression__other__person__model/dense_final/kernel/Read/ReadVariableOpReadVariableOp3regression__other__person__model/dense_final/kernel*
_output_shapes

:B*
dtype0
�
-regression__other__person__model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-regression__other__person__model/dense_1/bias
�
Aregression__other__person__model/dense_1/bias/Read/ReadVariableOpReadVariableOp-regression__other__person__model/dense_1/bias*
_output_shapes
:@*
dtype0
�
/regression__other__person__model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*@
shared_name1/regression__other__person__model/dense_1/kernel
�
Cregression__other__person__model/dense_1/kernel/Read/ReadVariableOpReadVariableOp/regression__other__person__model/dense_1/kernel*
_output_shapes
:	�@*
dtype0
�
-regression__other__person__model/dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-regression__other__person__model/dense_0/bias
�
Aregression__other__person__model/dense_0/bias/Read/ReadVariableOpReadVariableOp-regression__other__person__model/dense_0/bias*
_output_shapes	
:�*
dtype0
�
/regression__other__person__model/dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�4�*@
shared_name1/regression__other__person__model/dense_0/kernel
�
Cregression__other__person__model/dense_0/kernel/Read/ReadVariableOpReadVariableOp/regression__other__person__model/dense_0/kernel* 
_output_shapes
:
�4�*
dtype0
�
,regression__other__person__model/conv3R/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,regression__other__person__model/conv3R/bias
�
@regression__other__person__model/conv3R/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv3R/bias*
_output_shapes	
:�*
dtype0
�
.regression__other__person__model/conv3R/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*?
shared_name0.regression__other__person__model/conv3R/kernel
�
Bregression__other__person__model/conv3R/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv3R/kernel*'
_output_shapes
:@�*
dtype0
�
,regression__other__person__model/conv2R/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,regression__other__person__model/conv2R/bias
�
@regression__other__person__model/conv2R/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv2R/bias*
_output_shapes
:@*
dtype0
�
.regression__other__person__model/conv2R/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*?
shared_name0.regression__other__person__model/conv2R/kernel
�
Bregression__other__person__model/conv2R/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv2R/kernel*&
_output_shapes
: @*
dtype0
�
,regression__other__person__model/conv1R/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,regression__other__person__model/conv1R/bias
�
@regression__other__person__model/conv1R/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv1R/bias*
_output_shapes
: *
dtype0
�
.regression__other__person__model/conv1R/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *?
shared_name0.regression__other__person__model/conv1R/kernel
�
Bregression__other__person__model/conv1R/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv1R/kernel*&
_output_shapes
:  *
dtype0
�
,regression__other__person__model/conv0R/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,regression__other__person__model/conv0R/bias
�
@regression__other__person__model/conv0R/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv0R/bias*
_output_shapes
: *
dtype0
�
.regression__other__person__model/conv0R/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.regression__other__person__model/conv0R/kernel
�
Bregression__other__person__model/conv0R/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv0R/kernel*&
_output_shapes
: *
dtype0
�
,regression__other__person__model/conv3L/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,regression__other__person__model/conv3L/bias
�
@regression__other__person__model/conv3L/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv3L/bias*
_output_shapes	
:�*
dtype0
�
.regression__other__person__model/conv3L/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*?
shared_name0.regression__other__person__model/conv3L/kernel
�
Bregression__other__person__model/conv3L/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv3L/kernel*'
_output_shapes
:@�*
dtype0
�
,regression__other__person__model/conv2L/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,regression__other__person__model/conv2L/bias
�
@regression__other__person__model/conv2L/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv2L/bias*
_output_shapes
:@*
dtype0
�
.regression__other__person__model/conv2L/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*?
shared_name0.regression__other__person__model/conv2L/kernel
�
Bregression__other__person__model/conv2L/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv2L/kernel*&
_output_shapes
: @*
dtype0
�
,regression__other__person__model/conv1L/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,regression__other__person__model/conv1L/bias
�
@regression__other__person__model/conv1L/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv1L/bias*
_output_shapes
: *
dtype0
�
.regression__other__person__model/conv1L/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *?
shared_name0.regression__other__person__model/conv1L/kernel
�
Bregression__other__person__model/conv1L/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv1L/kernel*&
_output_shapes
:  *
dtype0
�
,regression__other__person__model/conv0L/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,regression__other__person__model/conv0L/bias
�
@regression__other__person__model/conv0L/bias/Read/ReadVariableOpReadVariableOp,regression__other__person__model/conv0L/bias*
_output_shapes
: *
dtype0
�
.regression__other__person__model/conv0L/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.regression__other__person__model/conv0L/kernel
�
Bregression__other__person__model/conv0L/kernel/Read/ReadVariableOpReadVariableOp.regression__other__person__model/conv0L/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ƨ
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
flatten
	pooling

dropout
conv_0L
conv_1L
conv_2L
conv_3L
conv_0R
conv_1R
conv_2R
conv_3R
dense_0
dense_1
final_dense
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15
(16
)17
*18
+19
,20
-21*
�
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15
(16
)17
*18
+19
,20
-21*
P
.0
/1
02
13
24
35
46
57
68
79
810* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
6
Btrace_0
Ctrace_1
Dtrace_2
Etrace_3* 
* 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses* 
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
X_random_generator* 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

kernel
bias
 __jit_compiled_convolution_op*
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

kernel
bias
 f_jit_compiled_convolution_op*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

kernel
bias
 m_jit_compiled_convolution_op*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

kernel
bias
 t_jit_compiled_convolution_op*
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

 kernel
!bias
 {_jit_compiled_convolution_op*
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses

"kernel
#bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

$kernel
%bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

&kernel
'bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�
activation

(kernel
)bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�
activation

*kernel
+bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

,kernel
-bias*
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�m�m�m�m�m�m� m�!m�"m�#m�$m�%m�&m�'m�(m�)m�*m�+m�,m�-m�v�v�v�v�v�v�v�v� v�!v�"v�#v�$v�%v�&v�'v�(v�)v�*v�+v�,v�-v�*

�serving_default* 
nh
VARIABLE_VALUE.regression__other__person__model/conv0L/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,regression__other__person__model/conv0L/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.regression__other__person__model/conv1L/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,regression__other__person__model/conv1L/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.regression__other__person__model/conv2L/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,regression__other__person__model/conv2L/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.regression__other__person__model/conv3L/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,regression__other__person__model/conv3L/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.regression__other__person__model/conv0R/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,regression__other__person__model/conv0R/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.regression__other__person__model/conv1R/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,regression__other__person__model/conv1R/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.regression__other__person__model/conv2R/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,regression__other__person__model/conv2R/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.regression__other__person__model/conv3R/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,regression__other__person__model/conv3R/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/regression__other__person__model/dense_0/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-regression__other__person__model/dense_0/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/regression__other__person__model/dense_1/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-regression__other__person__model/dense_1/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3regression__other__person__model/dense_final/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1regression__other__person__model/dense_final/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
j
0
	1

2
3
4
5
6
7
8
9
10
11
12
13*

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11* 
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11* 
* 

0
1*

0
1*
	
.0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
	
/0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
	
00* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
	
10* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

 0
!1*

 0
!1*
	
20* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

"0
#1*

"0
#1*
	
30* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

$0
%1*

$0
%1*
	
40* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

&0
'1*

&0
'1*
	
50* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

(0
)1*

(0
)1*
	
60* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 

*0
+1*

*0
+1*
	
70* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 

,0
-1*

,0
-1*
	
80* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
.0* 
* 
* 
* 
* 
* 
* 
	
/0* 
* 
* 
* 
* 
* 
* 
	
00* 
* 
* 
* 
* 
* 
* 
	
10* 
* 
* 
* 
* 
* 
* 
	
20* 
* 
* 
* 
* 
* 
* 
	
30* 
* 
* 
* 
* 
* 
* 
	
40* 
* 
* 
* 
* 
* 
* 
	
50* 
* 
* 
* 
* 


�0* 
* 
	
60* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 


�0* 
* 
	
70* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
	
80* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv0L/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv0L/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv1L/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv1L/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv2L/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv2L/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv3L/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv3L/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv0R/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv0R/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv1R/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv1R/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv2R/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv2R/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv3R/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv3R/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/regression__other__person__model/dense_0/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/regression__other__person__model/dense_0/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/regression__other__person__model/dense_1/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/regression__other__person__model/dense_1/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/regression__other__person__model/dense_final/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8Adam/regression__other__person__model/dense_final/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv0L/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv0L/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv1L/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv1L/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv2L/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv2L/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv3L/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv3L/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv0R/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv0R/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv1R/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv1R/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv2R/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv2R/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE5Adam/regression__other__person__model/conv3R/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/regression__other__person__model/conv3R/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/regression__other__person__model/dense_0/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/regression__other__person__model/dense_0/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/regression__other__person__model/dense_1/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/regression__other__person__model/dense_1/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE:Adam/regression__other__person__model/dense_final/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8Adam/regression__other__person__model/dense_final/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_1Placeholder*(
_output_shapes
:����������p*
dtype0*
shape:����������p
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1.regression__other__person__model/conv0L/kernel,regression__other__person__model/conv0L/bias.regression__other__person__model/conv1L/kernel,regression__other__person__model/conv1L/bias.regression__other__person__model/conv2L/kernel,regression__other__person__model/conv2L/bias.regression__other__person__model/conv3L/kernel,regression__other__person__model/conv3L/bias.regression__other__person__model/conv0R/kernel,regression__other__person__model/conv0R/bias.regression__other__person__model/conv1R/kernel,regression__other__person__model/conv1R/bias.regression__other__person__model/conv2R/kernel,regression__other__person__model/conv2R/bias.regression__other__person__model/conv3R/kernel,regression__other__person__model/conv3R/bias/regression__other__person__model/dense_0/kernel-regression__other__person__model/dense_0/bias/regression__other__person__model/dense_1/kernel-regression__other__person__model/dense_1/bias3regression__other__person__model/dense_final/kernel1regression__other__person__model/dense_final/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_9997
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameBregression__other__person__model/conv0L/kernel/Read/ReadVariableOp@regression__other__person__model/conv0L/bias/Read/ReadVariableOpBregression__other__person__model/conv1L/kernel/Read/ReadVariableOp@regression__other__person__model/conv1L/bias/Read/ReadVariableOpBregression__other__person__model/conv2L/kernel/Read/ReadVariableOp@regression__other__person__model/conv2L/bias/Read/ReadVariableOpBregression__other__person__model/conv3L/kernel/Read/ReadVariableOp@regression__other__person__model/conv3L/bias/Read/ReadVariableOpBregression__other__person__model/conv0R/kernel/Read/ReadVariableOp@regression__other__person__model/conv0R/bias/Read/ReadVariableOpBregression__other__person__model/conv1R/kernel/Read/ReadVariableOp@regression__other__person__model/conv1R/bias/Read/ReadVariableOpBregression__other__person__model/conv2R/kernel/Read/ReadVariableOp@regression__other__person__model/conv2R/bias/Read/ReadVariableOpBregression__other__person__model/conv3R/kernel/Read/ReadVariableOp@regression__other__person__model/conv3R/bias/Read/ReadVariableOpCregression__other__person__model/dense_0/kernel/Read/ReadVariableOpAregression__other__person__model/dense_0/bias/Read/ReadVariableOpCregression__other__person__model/dense_1/kernel/Read/ReadVariableOpAregression__other__person__model/dense_1/bias/Read/ReadVariableOpGregression__other__person__model/dense_final/kernel/Read/ReadVariableOpEregression__other__person__model/dense_final/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpIAdam/regression__other__person__model/conv0L/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv0L/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv1L/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv1L/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv2L/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv2L/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv3L/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv3L/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv0R/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv0R/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv1R/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv1R/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv2R/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv2R/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv3R/kernel/m/Read/ReadVariableOpGAdam/regression__other__person__model/conv3R/bias/m/Read/ReadVariableOpJAdam/regression__other__person__model/dense_0/kernel/m/Read/ReadVariableOpHAdam/regression__other__person__model/dense_0/bias/m/Read/ReadVariableOpJAdam/regression__other__person__model/dense_1/kernel/m/Read/ReadVariableOpHAdam/regression__other__person__model/dense_1/bias/m/Read/ReadVariableOpNAdam/regression__other__person__model/dense_final/kernel/m/Read/ReadVariableOpLAdam/regression__other__person__model/dense_final/bias/m/Read/ReadVariableOpIAdam/regression__other__person__model/conv0L/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv0L/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv1L/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv1L/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv2L/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv2L/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv3L/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv3L/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv0R/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv0R/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv1R/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv1R/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv2R/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv2R/bias/v/Read/ReadVariableOpIAdam/regression__other__person__model/conv3R/kernel/v/Read/ReadVariableOpGAdam/regression__other__person__model/conv3R/bias/v/Read/ReadVariableOpJAdam/regression__other__person__model/dense_0/kernel/v/Read/ReadVariableOpHAdam/regression__other__person__model/dense_0/bias/v/Read/ReadVariableOpJAdam/regression__other__person__model/dense_1/kernel/v/Read/ReadVariableOpHAdam/regression__other__person__model/dense_1/bias/v/Read/ReadVariableOpNAdam/regression__other__person__model/dense_final/kernel/v/Read/ReadVariableOpLAdam/regression__other__person__model/dense_final/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_11455
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename.regression__other__person__model/conv0L/kernel,regression__other__person__model/conv0L/bias.regression__other__person__model/conv1L/kernel,regression__other__person__model/conv1L/bias.regression__other__person__model/conv2L/kernel,regression__other__person__model/conv2L/bias.regression__other__person__model/conv3L/kernel,regression__other__person__model/conv3L/bias.regression__other__person__model/conv0R/kernel,regression__other__person__model/conv0R/bias.regression__other__person__model/conv1R/kernel,regression__other__person__model/conv1R/bias.regression__other__person__model/conv2R/kernel,regression__other__person__model/conv2R/bias.regression__other__person__model/conv3R/kernel,regression__other__person__model/conv3R/bias/regression__other__person__model/dense_0/kernel-regression__other__person__model/dense_0/bias/regression__other__person__model/dense_1/kernel-regression__other__person__model/dense_1/bias3regression__other__person__model/dense_final/kernel1regression__other__person__model/dense_final/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount5Adam/regression__other__person__model/conv0L/kernel/m3Adam/regression__other__person__model/conv0L/bias/m5Adam/regression__other__person__model/conv1L/kernel/m3Adam/regression__other__person__model/conv1L/bias/m5Adam/regression__other__person__model/conv2L/kernel/m3Adam/regression__other__person__model/conv2L/bias/m5Adam/regression__other__person__model/conv3L/kernel/m3Adam/regression__other__person__model/conv3L/bias/m5Adam/regression__other__person__model/conv0R/kernel/m3Adam/regression__other__person__model/conv0R/bias/m5Adam/regression__other__person__model/conv1R/kernel/m3Adam/regression__other__person__model/conv1R/bias/m5Adam/regression__other__person__model/conv2R/kernel/m3Adam/regression__other__person__model/conv2R/bias/m5Adam/regression__other__person__model/conv3R/kernel/m3Adam/regression__other__person__model/conv3R/bias/m6Adam/regression__other__person__model/dense_0/kernel/m4Adam/regression__other__person__model/dense_0/bias/m6Adam/regression__other__person__model/dense_1/kernel/m4Adam/regression__other__person__model/dense_1/bias/m:Adam/regression__other__person__model/dense_final/kernel/m8Adam/regression__other__person__model/dense_final/bias/m5Adam/regression__other__person__model/conv0L/kernel/v3Adam/regression__other__person__model/conv0L/bias/v5Adam/regression__other__person__model/conv1L/kernel/v3Adam/regression__other__person__model/conv1L/bias/v5Adam/regression__other__person__model/conv2L/kernel/v3Adam/regression__other__person__model/conv2L/bias/v5Adam/regression__other__person__model/conv3L/kernel/v3Adam/regression__other__person__model/conv3L/bias/v5Adam/regression__other__person__model/conv0R/kernel/v3Adam/regression__other__person__model/conv0R/bias/v5Adam/regression__other__person__model/conv1R/kernel/v3Adam/regression__other__person__model/conv1R/bias/v5Adam/regression__other__person__model/conv2R/kernel/v3Adam/regression__other__person__model/conv2R/bias/v5Adam/regression__other__person__model/conv3R/kernel/v3Adam/regression__other__person__model/conv3R/bias/v6Adam/regression__other__person__model/dense_0/kernel/v4Adam/regression__other__person__model/dense_0/bias/v6Adam/regression__other__person__model/dense_1/kernel/v4Adam/regression__other__person__model/dense_1/bias/v:Adam/regression__other__person__model/dense_final/kernel/v8Adam/regression__other__person__model/dense_final/bias/v*U
TinN
L2J*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_11684��
�
C
'__inference_dropout_layer_call_fn_10800

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8622h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������8@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8@:W S
/
_output_shapes
:���������8@
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_11170

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_8811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_6_10700s
Yregression__other__person__model_conv2r_kernel_regularizer_square_readvariableop_resource: @
identity��Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv2r_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv2R/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_4_10678s
Yregression__other__person__model_conv0r_kernel_regularizer_square_readvariableop_resource: 
identity��Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv0r_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv0R/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp
�	
`
A__inference_dropout_layer_call_and_return_conditional_losses_8992

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������BC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������B*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������Bo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������Bi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������BY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������B"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������B:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_9126

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������t C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������t *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������t w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������t q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������t a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������t "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������t :W S
/
_output_shapes
:���������t 
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_10840

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������8@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������8@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8@:W S
/
_output_shapes
:���������8@
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_10855

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������B[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������B"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������B:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
�
@__inference_conv3L_layer_call_and_return_conditional_losses_8642

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:�����������
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_dense_final_layer_call_fn_11196

inputs
unknown:B
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_final_layer_call_and_return_conditional_losses_8842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������B: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
`
'__inference_dropout_layer_call_fn_10805

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9094w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������8@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������8@
 
_user_specified_nameinputs
��
�
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_8915

inputs%
conv0l_8553: 
conv0l_8555: %
conv1l_8583:  
conv1l_8585: %
conv2l_8613: @
conv2l_8615:@&
conv3l_8643:@�
conv3l_8645:	�%
conv0r_8680: 
conv0r_8682: %
conv1r_8704:  
conv1r_8706: %
conv2r_8729: @
conv2r_8731:@&
conv3r_8754:@�
conv3r_8756:	� 
dense_0_8781:
�4�
dense_0_8783:	�
dense_1_8812:	�@
dense_1_8814:@"
dense_final_8843:B
dense_final_8845:
identity��conv0L/StatefulPartitionedCall�conv0R/StatefulPartitionedCall�conv1L/StatefulPartitionedCall�conv1R/StatefulPartitionedCall�conv2L/StatefulPartitionedCall�conv2R/StatefulPartitionedCall�conv3L/StatefulPartitionedCall�conv3R/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�#dense_final/StatefulPartitionedCall�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitVinputsConst:output:0split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_splitK
reshape/ShapeShapesplit:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapesplit:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������xM
reshape_1/ShapeShapesplit:output:1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapesplit:output:1 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
conv0L/StatefulPartitionedCallStatefulPartitionedCallreshape/Reshape:output:0conv0l_8553conv0l_8555*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0L_layer_call_and_return_conditional_losses_8552�
dropout/PartitionedCallPartitionedCall'conv0L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8563�
conv1L/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1l_8583conv1l_8585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1L_layer_call_and_return_conditional_losses_8582�
dropout/PartitionedCall_1PartitionedCall'conv1L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8592�
max_pooling/PartitionedCallPartitionedCall"dropout/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2L/StatefulPartitionedCallStatefulPartitionedCall$max_pooling/PartitionedCall:output:0conv2l_8613conv2l_8615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2L_layer_call_and_return_conditional_losses_8612�
dropout/PartitionedCall_2PartitionedCall'conv2L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8622�
max_pooling/PartitionedCall_1PartitionedCall"dropout/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3L/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_1:output:0conv3l_8643conv3l_8645*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3L_layer_call_and_return_conditional_losses_8642�
dropout/PartitionedCall_3PartitionedCall'conv3L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8652�
flatten/PartitionedCallPartitionedCall"dropout/PartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660�
conv0R/StatefulPartitionedCallStatefulPartitionedCallreshape_1/Reshape:output:0conv0r_8680conv0r_8682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0R_layer_call_and_return_conditional_losses_8679�
dropout/PartitionedCall_4PartitionedCall'conv0R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8563�
conv1R/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_4:output:0conv1r_8704conv1r_8706*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1R_layer_call_and_return_conditional_losses_8703�
dropout/PartitionedCall_5PartitionedCall'conv1R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8592�
max_pooling/PartitionedCall_2PartitionedCall"dropout/PartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_2:output:0conv2r_8729conv2r_8731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2R_layer_call_and_return_conditional_losses_8728�
dropout/PartitionedCall_6PartitionedCall'conv2R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8622�
max_pooling/PartitionedCall_3PartitionedCall"dropout/PartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_3:output:0conv3r_8754conv3r_8756*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3R_layer_call_and_return_conditional_losses_8753�
dropout/PartitionedCall_7PartitionedCall'conv3R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8652�
flatten/PartitionedCall_1PartitionedCall"dropout/PartitionedCall_7:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2 flatten/PartitionedCall:output:0"flatten/PartitionedCall_1:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
dense_0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0dense_0_8781dense_0_8783*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_0_layer_call_and_return_conditional_losses_8780[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2(dense_0/StatefulPartitionedCall:output:0split:output:2"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dropout/PartitionedCall_8PartitionedCallconcatenate_1/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8792�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_8:output:0dense_1_8812dense_1_8814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_8811[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2(dense_1/StatefulPartitionedCall:output:0split:output:2"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������B�
dropout/PartitionedCall_9PartitionedCallconcatenate_2/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������B* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8823�
#dense_final/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_9:output:0dense_final_8843dense_final_8845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_final_layer_call_and_return_conditional_losses_8842�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0l_8553*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1l_8583*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2l_8613*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3l_8643*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0r_8680*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1r_8704*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2r_8729*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3r_8754*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_8781* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_8812*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_8843*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp^conv0L/StatefulPartitionedCall^conv0R/StatefulPartitionedCall^conv1L/StatefulPartitionedCall^conv1R/StatefulPartitionedCall^conv2L/StatefulPartitionedCall^conv2R/StatefulPartitionedCall^conv3L/StatefulPartitionedCall^conv3R/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^dense_final/StatefulPartitionedCallQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2@
conv0L/StatefulPartitionedCallconv0L/StatefulPartitionedCall2@
conv0R/StatefulPartitionedCallconv0R/StatefulPartitionedCall2@
conv1L/StatefulPartitionedCallconv1L/StatefulPartitionedCall2@
conv1R/StatefulPartitionedCallconv1R/StatefulPartitionedCall2@
conv2L/StatefulPartitionedCallconv2L/StatefulPartitionedCall2@
conv2R/StatefulPartitionedCallconv2R/StatefulPartitionedCall2@
conv3L/StatefulPartitionedCallconv3L/StatefulPartitionedCall2@
conv3R/StatefulPartitionedCallconv3R/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������p
 
_user_specified_nameinputs
�
`
'__inference_dropout_layer_call_fn_10775

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������B* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8992o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������B`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������B22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_9997
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: @

unknown_12:@%

unknown_13:@�

unknown_14:	�

unknown_15:
�4�

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:B

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_8491o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������p
!
_user_specified_name	input_1
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_8811

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@o
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:���������@*
alpha%��u=�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_10755

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_10749

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_10830

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������v c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������v "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������v :W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
��
�>
!__inference__traced_restore_11684
file_prefixY
?assignvariableop_regression__other__person__model_conv0l_kernel: M
?assignvariableop_1_regression__other__person__model_conv0l_bias: [
Aassignvariableop_2_regression__other__person__model_conv1l_kernel:  M
?assignvariableop_3_regression__other__person__model_conv1l_bias: [
Aassignvariableop_4_regression__other__person__model_conv2l_kernel: @M
?assignvariableop_5_regression__other__person__model_conv2l_bias:@\
Aassignvariableop_6_regression__other__person__model_conv3l_kernel:@�N
?assignvariableop_7_regression__other__person__model_conv3l_bias:	�[
Aassignvariableop_8_regression__other__person__model_conv0r_kernel: M
?assignvariableop_9_regression__other__person__model_conv0r_bias: \
Bassignvariableop_10_regression__other__person__model_conv1r_kernel:  N
@assignvariableop_11_regression__other__person__model_conv1r_bias: \
Bassignvariableop_12_regression__other__person__model_conv2r_kernel: @N
@assignvariableop_13_regression__other__person__model_conv2r_bias:@]
Bassignvariableop_14_regression__other__person__model_conv3r_kernel:@�O
@assignvariableop_15_regression__other__person__model_conv3r_bias:	�W
Cassignvariableop_16_regression__other__person__model_dense_0_kernel:
�4�P
Aassignvariableop_17_regression__other__person__model_dense_0_bias:	�V
Cassignvariableop_18_regression__other__person__model_dense_1_kernel:	�@O
Aassignvariableop_19_regression__other__person__model_dense_1_bias:@Y
Gassignvariableop_20_regression__other__person__model_dense_final_kernel:BS
Eassignvariableop_21_regression__other__person__model_dense_final_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: #
assignvariableop_27_total: #
assignvariableop_28_count: c
Iassignvariableop_29_adam_regression__other__person__model_conv0l_kernel_m: U
Gassignvariableop_30_adam_regression__other__person__model_conv0l_bias_m: c
Iassignvariableop_31_adam_regression__other__person__model_conv1l_kernel_m:  U
Gassignvariableop_32_adam_regression__other__person__model_conv1l_bias_m: c
Iassignvariableop_33_adam_regression__other__person__model_conv2l_kernel_m: @U
Gassignvariableop_34_adam_regression__other__person__model_conv2l_bias_m:@d
Iassignvariableop_35_adam_regression__other__person__model_conv3l_kernel_m:@�V
Gassignvariableop_36_adam_regression__other__person__model_conv3l_bias_m:	�c
Iassignvariableop_37_adam_regression__other__person__model_conv0r_kernel_m: U
Gassignvariableop_38_adam_regression__other__person__model_conv0r_bias_m: c
Iassignvariableop_39_adam_regression__other__person__model_conv1r_kernel_m:  U
Gassignvariableop_40_adam_regression__other__person__model_conv1r_bias_m: c
Iassignvariableop_41_adam_regression__other__person__model_conv2r_kernel_m: @U
Gassignvariableop_42_adam_regression__other__person__model_conv2r_bias_m:@d
Iassignvariableop_43_adam_regression__other__person__model_conv3r_kernel_m:@�V
Gassignvariableop_44_adam_regression__other__person__model_conv3r_bias_m:	�^
Jassignvariableop_45_adam_regression__other__person__model_dense_0_kernel_m:
�4�W
Hassignvariableop_46_adam_regression__other__person__model_dense_0_bias_m:	�]
Jassignvariableop_47_adam_regression__other__person__model_dense_1_kernel_m:	�@V
Hassignvariableop_48_adam_regression__other__person__model_dense_1_bias_m:@`
Nassignvariableop_49_adam_regression__other__person__model_dense_final_kernel_m:BZ
Lassignvariableop_50_adam_regression__other__person__model_dense_final_bias_m:c
Iassignvariableop_51_adam_regression__other__person__model_conv0l_kernel_v: U
Gassignvariableop_52_adam_regression__other__person__model_conv0l_bias_v: c
Iassignvariableop_53_adam_regression__other__person__model_conv1l_kernel_v:  U
Gassignvariableop_54_adam_regression__other__person__model_conv1l_bias_v: c
Iassignvariableop_55_adam_regression__other__person__model_conv2l_kernel_v: @U
Gassignvariableop_56_adam_regression__other__person__model_conv2l_bias_v:@d
Iassignvariableop_57_adam_regression__other__person__model_conv3l_kernel_v:@�V
Gassignvariableop_58_adam_regression__other__person__model_conv3l_bias_v:	�c
Iassignvariableop_59_adam_regression__other__person__model_conv0r_kernel_v: U
Gassignvariableop_60_adam_regression__other__person__model_conv0r_bias_v: c
Iassignvariableop_61_adam_regression__other__person__model_conv1r_kernel_v:  U
Gassignvariableop_62_adam_regression__other__person__model_conv1r_bias_v: c
Iassignvariableop_63_adam_regression__other__person__model_conv2r_kernel_v: @U
Gassignvariableop_64_adam_regression__other__person__model_conv2r_bias_v:@d
Iassignvariableop_65_adam_regression__other__person__model_conv3r_kernel_v:@�V
Gassignvariableop_66_adam_regression__other__person__model_conv3r_bias_v:	�^
Jassignvariableop_67_adam_regression__other__person__model_dense_0_kernel_v:
�4�W
Hassignvariableop_68_adam_regression__other__person__model_dense_0_bias_v:	�]
Jassignvariableop_69_adam_regression__other__person__model_dense_1_kernel_v:	�@V
Hassignvariableop_70_adam_regression__other__person__model_dense_1_bias_v:@`
Nassignvariableop_71_adam_regression__other__person__model_dense_final_kernel_v:BZ
Lassignvariableop_72_adam_regression__other__person__model_dense_final_bias_v:
identity_74��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_8�AssignVariableOp_9�"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�!
value�!B�!JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�
value�B�JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp?assignvariableop_regression__other__person__model_conv0l_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp?assignvariableop_1_regression__other__person__model_conv0l_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpAassignvariableop_2_regression__other__person__model_conv1l_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp?assignvariableop_3_regression__other__person__model_conv1l_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpAassignvariableop_4_regression__other__person__model_conv2l_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp?assignvariableop_5_regression__other__person__model_conv2l_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpAassignvariableop_6_regression__other__person__model_conv3l_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp?assignvariableop_7_regression__other__person__model_conv3l_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpAassignvariableop_8_regression__other__person__model_conv0r_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp?assignvariableop_9_regression__other__person__model_conv0r_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpBassignvariableop_10_regression__other__person__model_conv1r_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp@assignvariableop_11_regression__other__person__model_conv1r_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpBassignvariableop_12_regression__other__person__model_conv2r_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp@assignvariableop_13_regression__other__person__model_conv2r_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpBassignvariableop_14_regression__other__person__model_conv3r_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp@assignvariableop_15_regression__other__person__model_conv3r_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpCassignvariableop_16_regression__other__person__model_dense_0_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpAassignvariableop_17_regression__other__person__model_dense_0_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpCassignvariableop_18_regression__other__person__model_dense_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpAassignvariableop_19_regression__other__person__model_dense_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpGassignvariableop_20_regression__other__person__model_dense_final_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpEassignvariableop_21_regression__other__person__model_dense_final_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpIassignvariableop_29_adam_regression__other__person__model_conv0l_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpGassignvariableop_30_adam_regression__other__person__model_conv0l_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpIassignvariableop_31_adam_regression__other__person__model_conv1l_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpGassignvariableop_32_adam_regression__other__person__model_conv1l_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpIassignvariableop_33_adam_regression__other__person__model_conv2l_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpGassignvariableop_34_adam_regression__other__person__model_conv2l_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpIassignvariableop_35_adam_regression__other__person__model_conv3l_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpGassignvariableop_36_adam_regression__other__person__model_conv3l_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpIassignvariableop_37_adam_regression__other__person__model_conv0r_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpGassignvariableop_38_adam_regression__other__person__model_conv0r_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpIassignvariableop_39_adam_regression__other__person__model_conv1r_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpGassignvariableop_40_adam_regression__other__person__model_conv1r_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpIassignvariableop_41_adam_regression__other__person__model_conv2r_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpGassignvariableop_42_adam_regression__other__person__model_conv2r_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpIassignvariableop_43_adam_regression__other__person__model_conv3r_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpGassignvariableop_44_adam_regression__other__person__model_conv3r_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpJassignvariableop_45_adam_regression__other__person__model_dense_0_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpHassignvariableop_46_adam_regression__other__person__model_dense_0_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpJassignvariableop_47_adam_regression__other__person__model_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpHassignvariableop_48_adam_regression__other__person__model_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpNassignvariableop_49_adam_regression__other__person__model_dense_final_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpLassignvariableop_50_adam_regression__other__person__model_dense_final_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpIassignvariableop_51_adam_regression__other__person__model_conv0l_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpGassignvariableop_52_adam_regression__other__person__model_conv0l_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpIassignvariableop_53_adam_regression__other__person__model_conv1l_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpGassignvariableop_54_adam_regression__other__person__model_conv1l_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpIassignvariableop_55_adam_regression__other__person__model_conv2l_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpGassignvariableop_56_adam_regression__other__person__model_conv2l_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpIassignvariableop_57_adam_regression__other__person__model_conv3l_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpGassignvariableop_58_adam_regression__other__person__model_conv3l_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpIassignvariableop_59_adam_regression__other__person__model_conv0r_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpGassignvariableop_60_adam_regression__other__person__model_conv0r_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpIassignvariableop_61_adam_regression__other__person__model_conv1r_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpGassignvariableop_62_adam_regression__other__person__model_conv1r_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpIassignvariableop_63_adam_regression__other__person__model_conv2r_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpGassignvariableop_64_adam_regression__other__person__model_conv2r_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpIassignvariableop_65_adam_regression__other__person__model_conv3r_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpGassignvariableop_66_adam_regression__other__person__model_conv3r_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpJassignvariableop_67_adam_regression__other__person__model_dense_0_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpHassignvariableop_68_adam_regression__other__person__model_dense_0_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpJassignvariableop_69_adam_regression__other__person__model_dense_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpHassignvariableop_70_adam_regression__other__person__model_dense_1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpNassignvariableop_71_adam_regression__other__person__model_dense_final_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpLassignvariableop_72_adam_regression__other__person__model_dense_final_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_74IdentityIdentity_73:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_74Identity_74:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
A__inference_conv3L_layer_call_and_return_conditional_losses_11031

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:�����������
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
F__inference_dense_final_layer_call_and_return_conditional_losses_11213

inputs0
matmul_readvariableop_resource:B-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:B*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:����������
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������B: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
a
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_9158

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������v C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������v *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������v w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������v q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������v a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������v "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������v :W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
֒
�
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9434

inputs%
conv0l_9290: 
conv0l_9292: %
conv1l_9296:  
conv1l_9298: %
conv2l_9303: @
conv2l_9305:@&
conv3l_9310:@�
conv3l_9312:	�%
conv0r_9317: 
conv0r_9319: %
conv1r_9323:  
conv1r_9325: %
conv2r_9330: @
conv2r_9332:@&
conv3r_9337:@�
conv3r_9339:	� 
dense_0_9346:
�4�
dense_0_9348:	�
dense_1_9354:	�@
dense_1_9356:@"
dense_final_9362:B
dense_final_9364:
identity��conv0L/StatefulPartitionedCall�conv0R/StatefulPartitionedCall�conv1L/StatefulPartitionedCall�conv1R/StatefulPartitionedCall�conv2L/StatefulPartitionedCall�conv2R/StatefulPartitionedCall�conv3L/StatefulPartitionedCall�conv3R/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�#dense_final/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout/StatefulPartitionedCall_1�!dropout/StatefulPartitionedCall_2�!dropout/StatefulPartitionedCall_3�!dropout/StatefulPartitionedCall_4�!dropout/StatefulPartitionedCall_5�!dropout/StatefulPartitionedCall_6�!dropout/StatefulPartitionedCall_7�!dropout/StatefulPartitionedCall_8�!dropout/StatefulPartitionedCall_9�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitVinputsConst:output:0split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_splitK
reshape/ShapeShapesplit:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapesplit:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������xM
reshape_1/ShapeShapesplit:output:1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapesplit:output:1 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
conv0L/StatefulPartitionedCallStatefulPartitionedCallreshape/Reshape:output:0conv0l_9290conv0l_9292*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0L_layer_call_and_return_conditional_losses_8552�
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv0L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9158�
conv1L/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1l_9296conv1l_9298*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1L_layer_call_and_return_conditional_losses_8582�
!dropout/StatefulPartitionedCall_1StatefulPartitionedCall'conv1L/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9126�
max_pooling/PartitionedCallPartitionedCall*dropout/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2L/StatefulPartitionedCallStatefulPartitionedCall$max_pooling/PartitionedCall:output:0conv2l_9303conv2l_9305*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2L_layer_call_and_return_conditional_losses_8612�
!dropout/StatefulPartitionedCall_2StatefulPartitionedCall'conv2L/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9094�
max_pooling/PartitionedCall_1PartitionedCall*dropout/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3L/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_1:output:0conv3l_9310conv3l_9312*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3L_layer_call_and_return_conditional_losses_8642�
!dropout/StatefulPartitionedCall_3StatefulPartitionedCall'conv3L/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9062�
flatten/PartitionedCallPartitionedCall*dropout/StatefulPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660�
conv0R/StatefulPartitionedCallStatefulPartitionedCallreshape_1/Reshape:output:0conv0r_9317conv0r_9319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0R_layer_call_and_return_conditional_losses_8679�
!dropout/StatefulPartitionedCall_4StatefulPartitionedCall'conv0R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_3*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9158�
conv1R/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_4:output:0conv1r_9323conv1r_9325*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1R_layer_call_and_return_conditional_losses_8703�
!dropout/StatefulPartitionedCall_5StatefulPartitionedCall'conv1R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_4*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9126�
max_pooling/PartitionedCall_2PartitionedCall*dropout/StatefulPartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_2:output:0conv2r_9330conv2r_9332*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2R_layer_call_and_return_conditional_losses_8728�
!dropout/StatefulPartitionedCall_6StatefulPartitionedCall'conv2R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_5*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9094�
max_pooling/PartitionedCall_3PartitionedCall*dropout/StatefulPartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_3:output:0conv3r_9337conv3r_9339*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3R_layer_call_and_return_conditional_losses_8753�
!dropout/StatefulPartitionedCall_7StatefulPartitionedCall'conv3R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9062�
flatten/PartitionedCall_1PartitionedCall*dropout/StatefulPartitionedCall_7:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2 flatten/PartitionedCall:output:0"flatten/PartitionedCall_1:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
dense_0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0dense_0_9346dense_0_9348*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_0_layer_call_and_return_conditional_losses_8780[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2(dense_0/StatefulPartitionedCall:output:0split:output:2"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
!dropout/StatefulPartitionedCall_8StatefulPartitionedCallconcatenate_1/concat:output:0"^dropout/StatefulPartitionedCall_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9024�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_8:output:0dense_1_9354dense_1_9356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_8811[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2(dense_1/StatefulPartitionedCall:output:0split:output:2"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������B�
!dropout/StatefulPartitionedCall_9StatefulPartitionedCallconcatenate_2/concat:output:0"^dropout/StatefulPartitionedCall_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������B* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8992�
#dense_final/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_9:output:0dense_final_9362dense_final_9364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_final_layer_call_and_return_conditional_losses_8842�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0l_9290*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1l_9296*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2l_9303*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3l_9310*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0r_9317*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1r_9323*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2r_9330*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3r_9337*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_9346* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_9354*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_9362*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv0L/StatefulPartitionedCall^conv0R/StatefulPartitionedCall^conv1L/StatefulPartitionedCall^conv1R/StatefulPartitionedCall^conv2L/StatefulPartitionedCall^conv2R/StatefulPartitionedCall^conv3L/StatefulPartitionedCall^conv3R/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^dense_final/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout/StatefulPartitionedCall_1"^dropout/StatefulPartitionedCall_2"^dropout/StatefulPartitionedCall_3"^dropout/StatefulPartitionedCall_4"^dropout/StatefulPartitionedCall_5"^dropout/StatefulPartitionedCall_6"^dropout/StatefulPartitionedCall_7"^dropout/StatefulPartitionedCall_8"^dropout/StatefulPartitionedCall_9Q^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2@
conv0L/StatefulPartitionedCallconv0L/StatefulPartitionedCall2@
conv0R/StatefulPartitionedCallconv0R/StatefulPartitionedCall2@
conv1L/StatefulPartitionedCallconv1L/StatefulPartitionedCall2@
conv1R/StatefulPartitionedCallconv1R/StatefulPartitionedCall2@
conv2L/StatefulPartitionedCallconv2L/StatefulPartitionedCall2@
conv2R/StatefulPartitionedCallconv2R/StatefulPartitionedCall2@
conv3L/StatefulPartitionedCallconv3L/StatefulPartitionedCall2@
conv3R/StatefulPartitionedCallconv3R/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout/StatefulPartitionedCall_1!dropout/StatefulPartitionedCall_12F
!dropout/StatefulPartitionedCall_2!dropout/StatefulPartitionedCall_22F
!dropout/StatefulPartitionedCall_3!dropout/StatefulPartitionedCall_32F
!dropout/StatefulPartitionedCall_4!dropout/StatefulPartitionedCall_42F
!dropout/StatefulPartitionedCall_5!dropout/StatefulPartitionedCall_52F
!dropout/StatefulPartitionedCall_6!dropout/StatefulPartitionedCall_62F
!dropout/StatefulPartitionedCall_7!dropout/StatefulPartitionedCall_72F
!dropout/StatefulPartitionedCall_8!dropout/StatefulPartitionedCall_82F
!dropout/StatefulPartitionedCall_9!dropout/StatefulPartitionedCall_92�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������p
 
_user_specified_nameinputs
��
�
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10557

inputs?
%conv0l_conv2d_readvariableop_resource: 4
&conv0l_biasadd_readvariableop_resource: ?
%conv1l_conv2d_readvariableop_resource:  4
&conv1l_biasadd_readvariableop_resource: ?
%conv2l_conv2d_readvariableop_resource: @4
&conv2l_biasadd_readvariableop_resource:@@
%conv3l_conv2d_readvariableop_resource:@�5
&conv3l_biasadd_readvariableop_resource:	�?
%conv0r_conv2d_readvariableop_resource: 4
&conv0r_biasadd_readvariableop_resource: ?
%conv1r_conv2d_readvariableop_resource:  4
&conv1r_biasadd_readvariableop_resource: ?
%conv2r_conv2d_readvariableop_resource: @4
&conv2r_biasadd_readvariableop_resource:@@
%conv3r_conv2d_readvariableop_resource:@�5
&conv3r_biasadd_readvariableop_resource:	�:
&dense_0_matmul_readvariableop_resource:
�4�6
'dense_0_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@<
*dense_final_matmul_readvariableop_resource:B9
+dense_final_biasadd_readvariableop_resource:
identity��conv0L/BiasAdd/ReadVariableOp�conv0L/Conv2D/ReadVariableOp�conv0R/BiasAdd/ReadVariableOp�conv0R/Conv2D/ReadVariableOp�conv1L/BiasAdd/ReadVariableOp�conv1L/Conv2D/ReadVariableOp�conv1R/BiasAdd/ReadVariableOp�conv1R/Conv2D/ReadVariableOp�conv2L/BiasAdd/ReadVariableOp�conv2L/Conv2D/ReadVariableOp�conv2R/BiasAdd/ReadVariableOp�conv2R/Conv2D/ReadVariableOp�conv3L/BiasAdd/ReadVariableOp�conv3L/Conv2D/ReadVariableOp�conv3R/BiasAdd/ReadVariableOp�conv3R/Conv2D/ReadVariableOp�dense_0/BiasAdd/ReadVariableOp�dense_0/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�"dense_final/BiasAdd/ReadVariableOp�!dense_final/MatMul/ReadVariableOp�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitVinputsConst:output:0split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_splitK
reshape/ShapeShapesplit:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapesplit:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������xM
reshape_1/ShapeShapesplit:output:1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapesplit:output:1 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
conv0L/Conv2D/ReadVariableOpReadVariableOp%conv0l_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv0L/Conv2DConv2Dreshape/Reshape:output:0$conv0L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
�
conv0L/BiasAdd/ReadVariableOpReadVariableOp&conv0l_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv0L/BiasAddBiasAddconv0L/Conv2D:output:0%conv0L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v f
conv0L/ReluReluconv0L/BiasAdd:output:0*
T0*/
_output_shapes
:���������v Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout/MulMulconv0L/Relu:activations:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������v ^
dropout/dropout/ShapeShapeconv0L/Relu:activations:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������v *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������v �
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������v �
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������v �
conv1L/Conv2D/ReadVariableOpReadVariableOp%conv1l_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv1L/Conv2DConv2Ddropout/dropout/Mul_1:z:0$conv1L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
�
conv1L/BiasAdd/ReadVariableOpReadVariableOp&conv1l_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv1L/BiasAddBiasAddconv1L/Conv2D:output:0%conv1L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t f
conv1L/ReluReluconv1L/BiasAdd:output:0*
T0*/
_output_shapes
:���������t \
dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_1/MulMulconv1L/Relu:activations:0 dropout/dropout_1/Const:output:0*
T0*/
_output_shapes
:���������t `
dropout/dropout_1/ShapeShapeconv1L/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_1/random_uniform/RandomUniformRandomUniform dropout/dropout_1/Shape:output:0*
T0*/
_output_shapes
:���������t *
dtype0e
 dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_1/GreaterEqualGreaterEqual7dropout/dropout_1/random_uniform/RandomUniform:output:0)dropout/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������t �
dropout/dropout_1/CastCast"dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������t �
dropout/dropout_1/Mul_1Muldropout/dropout_1/Mul:z:0dropout/dropout_1/Cast:y:0*
T0*/
_output_shapes
:���������t �
max_pooling/MaxPoolMaxPooldropout/dropout_1/Mul_1:z:0*/
_output_shapes
:���������: *
ksize
*
paddingVALID*
strides
�
conv2L/Conv2D/ReadVariableOpReadVariableOp%conv2l_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2L/Conv2DConv2Dmax_pooling/MaxPool:output:0$conv2L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
�
conv2L/BiasAdd/ReadVariableOpReadVariableOp&conv2l_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2L/BiasAddBiasAddconv2L/Conv2D:output:0%conv2L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@f
conv2L/ReluReluconv2L/BiasAdd:output:0*
T0*/
_output_shapes
:���������8@\
dropout/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_2/MulMulconv2L/Relu:activations:0 dropout/dropout_2/Const:output:0*
T0*/
_output_shapes
:���������8@`
dropout/dropout_2/ShapeShapeconv2L/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_2/random_uniform/RandomUniformRandomUniform dropout/dropout_2/Shape:output:0*
T0*/
_output_shapes
:���������8@*
dtype0e
 dropout/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_2/GreaterEqualGreaterEqual7dropout/dropout_2/random_uniform/RandomUniform:output:0)dropout/dropout_2/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������8@�
dropout/dropout_2/CastCast"dropout/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������8@�
dropout/dropout_2/Mul_1Muldropout/dropout_2/Mul:z:0dropout/dropout_2/Cast:y:0*
T0*/
_output_shapes
:���������8@�
max_pooling/MaxPool_1MaxPooldropout/dropout_2/Mul_1:z:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv3L/Conv2D/ReadVariableOpReadVariableOp%conv3l_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv3L/Conv2DConv2Dmax_pooling/MaxPool_1:output:0$conv3L/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv3L/BiasAdd/ReadVariableOpReadVariableOp&conv3l_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3L/BiasAddBiasAddconv3L/Conv2D:output:0%conv3L/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv3L/ReluReluconv3L/BiasAdd:output:0*
T0*0
_output_shapes
:����������\
dropout/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_3/MulMulconv3L/Relu:activations:0 dropout/dropout_3/Const:output:0*
T0*0
_output_shapes
:����������`
dropout/dropout_3/ShapeShapeconv3L/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_3/random_uniform/RandomUniformRandomUniform dropout/dropout_3/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0e
 dropout/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_3/GreaterEqualGreaterEqual7dropout/dropout_3/random_uniform/RandomUniform:output:0)dropout/dropout_3/GreaterEqual/y:output:0*
T0*0
_output_shapes
:�����������
dropout/dropout_3/CastCast"dropout/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:�����������
dropout/dropout_3/Mul_1Muldropout/dropout_3/Mul:z:0dropout/dropout_3/Cast:y:0*
T0*0
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapedropout/dropout_3/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
conv0R/Conv2D/ReadVariableOpReadVariableOp%conv0r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv0R/Conv2DConv2Dreshape_1/Reshape:output:0$conv0R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
�
conv0R/BiasAdd/ReadVariableOpReadVariableOp&conv0r_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv0R/BiasAddBiasAddconv0R/Conv2D:output:0%conv0R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v f
conv0R/ReluReluconv0R/BiasAdd:output:0*
T0*/
_output_shapes
:���������v \
dropout/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_4/MulMulconv0R/Relu:activations:0 dropout/dropout_4/Const:output:0*
T0*/
_output_shapes
:���������v `
dropout/dropout_4/ShapeShapeconv0R/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_4/random_uniform/RandomUniformRandomUniform dropout/dropout_4/Shape:output:0*
T0*/
_output_shapes
:���������v *
dtype0e
 dropout/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_4/GreaterEqualGreaterEqual7dropout/dropout_4/random_uniform/RandomUniform:output:0)dropout/dropout_4/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������v �
dropout/dropout_4/CastCast"dropout/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������v �
dropout/dropout_4/Mul_1Muldropout/dropout_4/Mul:z:0dropout/dropout_4/Cast:y:0*
T0*/
_output_shapes
:���������v �
conv1R/Conv2D/ReadVariableOpReadVariableOp%conv1r_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv1R/Conv2DConv2Ddropout/dropout_4/Mul_1:z:0$conv1R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
�
conv1R/BiasAdd/ReadVariableOpReadVariableOp&conv1r_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv1R/BiasAddBiasAddconv1R/Conv2D:output:0%conv1R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t f
conv1R/ReluReluconv1R/BiasAdd:output:0*
T0*/
_output_shapes
:���������t \
dropout/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_5/MulMulconv1R/Relu:activations:0 dropout/dropout_5/Const:output:0*
T0*/
_output_shapes
:���������t `
dropout/dropout_5/ShapeShapeconv1R/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_5/random_uniform/RandomUniformRandomUniform dropout/dropout_5/Shape:output:0*
T0*/
_output_shapes
:���������t *
dtype0e
 dropout/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_5/GreaterEqualGreaterEqual7dropout/dropout_5/random_uniform/RandomUniform:output:0)dropout/dropout_5/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������t �
dropout/dropout_5/CastCast"dropout/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������t �
dropout/dropout_5/Mul_1Muldropout/dropout_5/Mul:z:0dropout/dropout_5/Cast:y:0*
T0*/
_output_shapes
:���������t �
max_pooling/MaxPool_2MaxPooldropout/dropout_5/Mul_1:z:0*/
_output_shapes
:���������: *
ksize
*
paddingVALID*
strides
�
conv2R/Conv2D/ReadVariableOpReadVariableOp%conv2r_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2R/Conv2DConv2Dmax_pooling/MaxPool_2:output:0$conv2R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
�
conv2R/BiasAdd/ReadVariableOpReadVariableOp&conv2r_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2R/BiasAddBiasAddconv2R/Conv2D:output:0%conv2R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@f
conv2R/ReluReluconv2R/BiasAdd:output:0*
T0*/
_output_shapes
:���������8@\
dropout/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_6/MulMulconv2R/Relu:activations:0 dropout/dropout_6/Const:output:0*
T0*/
_output_shapes
:���������8@`
dropout/dropout_6/ShapeShapeconv2R/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_6/random_uniform/RandomUniformRandomUniform dropout/dropout_6/Shape:output:0*
T0*/
_output_shapes
:���������8@*
dtype0e
 dropout/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_6/GreaterEqualGreaterEqual7dropout/dropout_6/random_uniform/RandomUniform:output:0)dropout/dropout_6/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������8@�
dropout/dropout_6/CastCast"dropout/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������8@�
dropout/dropout_6/Mul_1Muldropout/dropout_6/Mul:z:0dropout/dropout_6/Cast:y:0*
T0*/
_output_shapes
:���������8@�
max_pooling/MaxPool_3MaxPooldropout/dropout_6/Mul_1:z:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv3R/Conv2D/ReadVariableOpReadVariableOp%conv3r_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv3R/Conv2DConv2Dmax_pooling/MaxPool_3:output:0$conv3R/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv3R/BiasAdd/ReadVariableOpReadVariableOp&conv3r_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3R/BiasAddBiasAddconv3R/Conv2D:output:0%conv3R/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv3R/ReluReluconv3R/BiasAdd:output:0*
T0*0
_output_shapes
:����������\
dropout/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_7/MulMulconv3R/Relu:activations:0 dropout/dropout_7/Const:output:0*
T0*0
_output_shapes
:����������`
dropout/dropout_7/ShapeShapeconv3R/Relu:activations:0*
T0*
_output_shapes
:�
.dropout/dropout_7/random_uniform/RandomUniformRandomUniform dropout/dropout_7/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0e
 dropout/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_7/GreaterEqualGreaterEqual7dropout/dropout_7/random_uniform/RandomUniform:output:0)dropout/dropout_7/GreaterEqual/y:output:0*
T0*0
_output_shapes
:�����������
dropout/dropout_7/CastCast"dropout/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:�����������
dropout/dropout_7/Mul_1Muldropout/dropout_7/Mul:z:0dropout/dropout_7/Cast:y:0*
T0*0
_output_shapes
:����������`
flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/Reshape_1Reshapedropout/dropout_7/Mul_1:z:0flatten/Const_1:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0flatten/Reshape_1:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
dense_0/MatMulMatMulconcatenate/concat:output:0%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������~
dense_0/leaky_re_lu/LeakyRelu	LeakyReludense_0/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%��u=[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2+dense_0/leaky_re_lu/LeakyRelu:activations:0split:output:2"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������\
dropout/dropout_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_8/MulMulconcatenate_1/concat:output:0 dropout/dropout_8/Const:output:0*
T0*(
_output_shapes
:����������d
dropout/dropout_8/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:�
.dropout/dropout_8/random_uniform/RandomUniformRandomUniform dropout/dropout_8/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0e
 dropout/dropout_8/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_8/GreaterEqualGreaterEqual7dropout/dropout_8/random_uniform/RandomUniform:output:0)dropout/dropout_8/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout/dropout_8/CastCast"dropout/dropout_8/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout/dropout_8/Mul_1Muldropout/dropout_8/Mul:z:0dropout/dropout_8/Cast:y:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_1/MatMulMatMuldropout/dropout_8/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
dense_1/leaky_re_lu_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:���������@*
alpha%��u=[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2-dense_1/leaky_re_lu_1/LeakyRelu:activations:0split:output:2"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������B\
dropout/dropout_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?�
dropout/dropout_9/MulMulconcatenate_2/concat:output:0 dropout/dropout_9/Const:output:0*
T0*'
_output_shapes
:���������Bd
dropout/dropout_9/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:�
.dropout/dropout_9/random_uniform/RandomUniformRandomUniform dropout/dropout_9/Shape:output:0*
T0*'
_output_shapes
:���������B*
dtype0e
 dropout/dropout_9/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/dropout_9/GreaterEqualGreaterEqual7dropout/dropout_9/random_uniform/RandomUniform:output:0)dropout/dropout_9/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������B�
dropout/dropout_9/CastCast"dropout/dropout_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������B�
dropout/dropout_9/Mul_1Muldropout/dropout_9/Mul:z:0dropout/dropout_9/Cast:y:0*
T0*'
_output_shapes
:���������B�
!dense_final/MatMul/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
dense_final/MatMulMatMuldropout/dropout_9/Mul_1:z:0)dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_final/BiasAdd/ReadVariableOpReadVariableOp+dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_final/BiasAddBiasAdddense_final/MatMul:product:0*dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
dense_final/SigmoidSigmoiddense_final/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv0l_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv1l_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2l_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3l_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv0r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv1r_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2r_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3r_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentitydense_final/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv0L/BiasAdd/ReadVariableOp^conv0L/Conv2D/ReadVariableOp^conv0R/BiasAdd/ReadVariableOp^conv0R/Conv2D/ReadVariableOp^conv1L/BiasAdd/ReadVariableOp^conv1L/Conv2D/ReadVariableOp^conv1R/BiasAdd/ReadVariableOp^conv1R/Conv2D/ReadVariableOp^conv2L/BiasAdd/ReadVariableOp^conv2L/Conv2D/ReadVariableOp^conv2R/BiasAdd/ReadVariableOp^conv2R/Conv2D/ReadVariableOp^conv3L/BiasAdd/ReadVariableOp^conv3L/Conv2D/ReadVariableOp^conv3R/BiasAdd/ReadVariableOp^conv3R/Conv2D/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^dense_final/BiasAdd/ReadVariableOp"^dense_final/MatMul/ReadVariableOpQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2>
conv0L/BiasAdd/ReadVariableOpconv0L/BiasAdd/ReadVariableOp2<
conv0L/Conv2D/ReadVariableOpconv0L/Conv2D/ReadVariableOp2>
conv0R/BiasAdd/ReadVariableOpconv0R/BiasAdd/ReadVariableOp2<
conv0R/Conv2D/ReadVariableOpconv0R/Conv2D/ReadVariableOp2>
conv1L/BiasAdd/ReadVariableOpconv1L/BiasAdd/ReadVariableOp2<
conv1L/Conv2D/ReadVariableOpconv1L/Conv2D/ReadVariableOp2>
conv1R/BiasAdd/ReadVariableOpconv1R/BiasAdd/ReadVariableOp2<
conv1R/Conv2D/ReadVariableOpconv1R/Conv2D/ReadVariableOp2>
conv2L/BiasAdd/ReadVariableOpconv2L/BiasAdd/ReadVariableOp2<
conv2L/Conv2D/ReadVariableOpconv2L/Conv2D/ReadVariableOp2>
conv2R/BiasAdd/ReadVariableOpconv2R/BiasAdd/ReadVariableOp2<
conv2R/Conv2D/ReadVariableOpconv2R/Conv2D/ReadVariableOp2>
conv3L/BiasAdd/ReadVariableOpconv3L/BiasAdd/ReadVariableOp2<
conv3L/Conv2D/ReadVariableOpconv3L/Conv2D/ReadVariableOp2>
conv3R/BiasAdd/ReadVariableOpconv3R/BiasAdd/ReadVariableOp2<
conv3R/Conv2D/ReadVariableOpconv3R/Conv2D/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"dense_final/BiasAdd/ReadVariableOp"dense_final/BiasAdd/ReadVariableOp2F
!dense_final/MatMul/ReadVariableOp!dense_final/MatMul/ReadVariableOp2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������p
 
_user_specified_nameinputs
�
�
?__inference_regression__other__person__model_layer_call_fn_9530
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: @

unknown_12:@%

unknown_13:@�

unknown_14:	�

unknown_15:
�4�

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:B

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *c
f^R\
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9434o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������p
!
_user_specified_name	input_1
�
�
A__inference_dense_0_layer_call_and_return_conditional_losses_8780

inputs2
matmul_readvariableop_resource:
�4�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������n
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������*
alpha%��u=�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������4
 
_user_specified_nameinputs
�
�
@__inference_conv1L_layer_call_and_return_conditional_losses_8582

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������t �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������t �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������v : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8792

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
'__inference_dropout_layer_call_fn_10795

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9062x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_conv2L_layer_call_and_return_conditional_losses_8612

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������8@�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������8@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������: 
 
_user_specified_nameinputs
�
b
F__inference_max_pooling_layer_call_and_return_conditional_losses_10765

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
@__inference_regression__other__person__model_layer_call_fn_10046

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: @

unknown_12:@%

unknown_13:@�

unknown_14:	�

unknown_15:
�4�

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:B

unknown_20:
identity��StatefulPartitionedCall�
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *c
f^R\
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_8915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������p
 
_user_specified_nameinputs
�	
`
A__inference_dropout_layer_call_and_return_conditional_losses_9024

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_conv2R_layer_call_fn_11092

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2R_layer_call_and_return_conditional_losses_8728w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������8@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������: 
 
_user_specified_nameinputs
�
�
@__inference_regression__other__person__model_layer_call_fn_10095

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: @

unknown_12:@%

unknown_13:@�

unknown_14:	�

unknown_15:
�4�

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:B

unknown_20:
identity��StatefulPartitionedCall�
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *c
f^R\
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9434o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������p
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_10835

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������t c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������t "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������t :W S
/
_output_shapes
:���������t 
 
_user_specified_nameinputs
�
�
@__inference_conv0R_layer_call_and_return_conditional_losses_8679

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������v �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������v �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������x
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_10903

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������8@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������8@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������8@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������8@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������8@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������8@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8@:W S
/
_output_shapes
:���������8@
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_10845

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8652

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_conv3L_layer_call_fn_11014

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3L_layer_call_and_return_conditional_losses_8642x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_dense_final_layer_call_and_return_conditional_losses_8842

inputs0
matmul_readvariableop_resource:B-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:B*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:����������
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������B: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
�
&__inference_conv1R_layer_call_fn_11066

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1R_layer_call_and_return_conditional_losses_8703w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������t `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������v : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_10891

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_9062

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_conv0R_layer_call_fn_11040

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0R_layer_call_and_return_conditional_losses_8679w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������v `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������x
 
_user_specified_nameinputs
�
�
&__inference_conv1L_layer_call_fn_10962

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1L_layer_call_and_return_conditional_losses_8582w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������t `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������v : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
�
'__inference_dense_0_layer_call_fn_11144

inputs
unknown:
�4�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_0_layer_call_and_return_conditional_losses_8780p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������4: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������4
 
_user_specified_nameinputs
��
�.
__inference__traced_save_11455
file_prefixM
Isavev2_regression__other__person__model_conv0l_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv0l_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv1l_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv1l_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv2l_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv2l_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv3l_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv3l_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv0r_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv0r_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv1r_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv1r_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv2r_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv2r_bias_read_readvariableopM
Isavev2_regression__other__person__model_conv3r_kernel_read_readvariableopK
Gsavev2_regression__other__person__model_conv3r_bias_read_readvariableopN
Jsavev2_regression__other__person__model_dense_0_kernel_read_readvariableopL
Hsavev2_regression__other__person__model_dense_0_bias_read_readvariableopN
Jsavev2_regression__other__person__model_dense_1_kernel_read_readvariableopL
Hsavev2_regression__other__person__model_dense_1_bias_read_readvariableopR
Nsavev2_regression__other__person__model_dense_final_kernel_read_readvariableopP
Lsavev2_regression__other__person__model_dense_final_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv0l_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv0l_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv1l_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv1l_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv2l_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv2l_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv3l_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv3l_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv0r_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv0r_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv1r_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv1r_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv2r_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv2r_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv3r_kernel_m_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv3r_bias_m_read_readvariableopU
Qsavev2_adam_regression__other__person__model_dense_0_kernel_m_read_readvariableopS
Osavev2_adam_regression__other__person__model_dense_0_bias_m_read_readvariableopU
Qsavev2_adam_regression__other__person__model_dense_1_kernel_m_read_readvariableopS
Osavev2_adam_regression__other__person__model_dense_1_bias_m_read_readvariableopY
Usavev2_adam_regression__other__person__model_dense_final_kernel_m_read_readvariableopW
Ssavev2_adam_regression__other__person__model_dense_final_bias_m_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv0l_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv0l_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv1l_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv1l_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv2l_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv2l_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv3l_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv3l_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv0r_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv0r_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv1r_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv1r_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv2r_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv2r_bias_v_read_readvariableopT
Psavev2_adam_regression__other__person__model_conv3r_kernel_v_read_readvariableopR
Nsavev2_adam_regression__other__person__model_conv3r_bias_v_read_readvariableopU
Qsavev2_adam_regression__other__person__model_dense_0_kernel_v_read_readvariableopS
Osavev2_adam_regression__other__person__model_dense_0_bias_v_read_readvariableopU
Qsavev2_adam_regression__other__person__model_dense_1_kernel_v_read_readvariableopS
Osavev2_adam_regression__other__person__model_dense_1_bias_v_read_readvariableopY
Usavev2_adam_regression__other__person__model_dense_final_kernel_v_read_readvariableopW
Ssavev2_adam_regression__other__person__model_dense_final_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�!
value�!B�!JB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*�
value�B�JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Isavev2_regression__other__person__model_conv0l_kernel_read_readvariableopGsavev2_regression__other__person__model_conv0l_bias_read_readvariableopIsavev2_regression__other__person__model_conv1l_kernel_read_readvariableopGsavev2_regression__other__person__model_conv1l_bias_read_readvariableopIsavev2_regression__other__person__model_conv2l_kernel_read_readvariableopGsavev2_regression__other__person__model_conv2l_bias_read_readvariableopIsavev2_regression__other__person__model_conv3l_kernel_read_readvariableopGsavev2_regression__other__person__model_conv3l_bias_read_readvariableopIsavev2_regression__other__person__model_conv0r_kernel_read_readvariableopGsavev2_regression__other__person__model_conv0r_bias_read_readvariableopIsavev2_regression__other__person__model_conv1r_kernel_read_readvariableopGsavev2_regression__other__person__model_conv1r_bias_read_readvariableopIsavev2_regression__other__person__model_conv2r_kernel_read_readvariableopGsavev2_regression__other__person__model_conv2r_bias_read_readvariableopIsavev2_regression__other__person__model_conv3r_kernel_read_readvariableopGsavev2_regression__other__person__model_conv3r_bias_read_readvariableopJsavev2_regression__other__person__model_dense_0_kernel_read_readvariableopHsavev2_regression__other__person__model_dense_0_bias_read_readvariableopJsavev2_regression__other__person__model_dense_1_kernel_read_readvariableopHsavev2_regression__other__person__model_dense_1_bias_read_readvariableopNsavev2_regression__other__person__model_dense_final_kernel_read_readvariableopLsavev2_regression__other__person__model_dense_final_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopPsavev2_adam_regression__other__person__model_conv0l_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv0l_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv1l_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv1l_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv2l_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv2l_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv3l_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv3l_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv0r_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv0r_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv1r_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv1r_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv2r_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv2r_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv3r_kernel_m_read_readvariableopNsavev2_adam_regression__other__person__model_conv3r_bias_m_read_readvariableopQsavev2_adam_regression__other__person__model_dense_0_kernel_m_read_readvariableopOsavev2_adam_regression__other__person__model_dense_0_bias_m_read_readvariableopQsavev2_adam_regression__other__person__model_dense_1_kernel_m_read_readvariableopOsavev2_adam_regression__other__person__model_dense_1_bias_m_read_readvariableopUsavev2_adam_regression__other__person__model_dense_final_kernel_m_read_readvariableopSsavev2_adam_regression__other__person__model_dense_final_bias_m_read_readvariableopPsavev2_adam_regression__other__person__model_conv0l_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv0l_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv1l_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv1l_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv2l_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv2l_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv3l_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv3l_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv0r_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv0r_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv1r_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv1r_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv2r_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv2r_bias_v_read_readvariableopPsavev2_adam_regression__other__person__model_conv3r_kernel_v_read_readvariableopNsavev2_adam_regression__other__person__model_conv3r_bias_v_read_readvariableopQsavev2_adam_regression__other__person__model_dense_0_kernel_v_read_readvariableopOsavev2_adam_regression__other__person__model_dense_0_bias_v_read_readvariableopQsavev2_adam_regression__other__person__model_dense_1_kernel_v_read_readvariableopOsavev2_adam_regression__other__person__model_dense_1_bias_v_read_readvariableopUsavev2_adam_regression__other__person__model_dense_final_kernel_v_read_readvariableopSsavev2_adam_regression__other__person__model_dense_final_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@�:�: : :  : : @:@:@�:�:
�4�:�:	�@:@:B:: : : : : : : : : :  : : @:@:@�:�: : :  : : @:@:@�:�:
�4�:�:	�@:@:B:: : :  : : @:@:@�:�: : :  : : @:@:@�:�:
�4�:�:	�@:@:B:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:,	(
&
_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:&"
 
_output_shapes
:
�4�:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:B: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:-$)
'
_output_shapes
:@�:!%

_output_shapes	
:�:,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
:  : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:-,)
'
_output_shapes
:@�:!-

_output_shapes	
:�:&."
 
_output_shapes
:
�4�:!/

_output_shapes	
:�:%0!

_output_shapes
:	�@: 1

_output_shapes
:@:$2 

_output_shapes

:B: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: :,6(
&
_output_shapes
:  : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@:-:)
'
_output_shapes
:@�:!;

_output_shapes	
:�:,<(
&
_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
:  : ?

_output_shapes
: :,@(
&
_output_shapes
: @: A

_output_shapes
:@:-B)
'
_output_shapes
:@�:!C

_output_shapes	
:�:&D"
 
_output_shapes
:
�4�:!E

_output_shapes	
:�:%F!

_output_shapes
:	�@: G

_output_shapes
:@:$H 

_output_shapes

:B: I

_output_shapes
::J

_output_shapes
: 
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8622

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������8@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������8@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8@:W S
/
_output_shapes
:���������8@
 
_user_specified_nameinputs
��
�
__inference__wrapped_model_8491
input_1`
Fregression__other__person__model_conv0l_conv2d_readvariableop_resource: U
Gregression__other__person__model_conv0l_biasadd_readvariableop_resource: `
Fregression__other__person__model_conv1l_conv2d_readvariableop_resource:  U
Gregression__other__person__model_conv1l_biasadd_readvariableop_resource: `
Fregression__other__person__model_conv2l_conv2d_readvariableop_resource: @U
Gregression__other__person__model_conv2l_biasadd_readvariableop_resource:@a
Fregression__other__person__model_conv3l_conv2d_readvariableop_resource:@�V
Gregression__other__person__model_conv3l_biasadd_readvariableop_resource:	�`
Fregression__other__person__model_conv0r_conv2d_readvariableop_resource: U
Gregression__other__person__model_conv0r_biasadd_readvariableop_resource: `
Fregression__other__person__model_conv1r_conv2d_readvariableop_resource:  U
Gregression__other__person__model_conv1r_biasadd_readvariableop_resource: `
Fregression__other__person__model_conv2r_conv2d_readvariableop_resource: @U
Gregression__other__person__model_conv2r_biasadd_readvariableop_resource:@a
Fregression__other__person__model_conv3r_conv2d_readvariableop_resource:@�V
Gregression__other__person__model_conv3r_biasadd_readvariableop_resource:	�[
Gregression__other__person__model_dense_0_matmul_readvariableop_resource:
�4�W
Hregression__other__person__model_dense_0_biasadd_readvariableop_resource:	�Z
Gregression__other__person__model_dense_1_matmul_readvariableop_resource:	�@V
Hregression__other__person__model_dense_1_biasadd_readvariableop_resource:@]
Kregression__other__person__model_dense_final_matmul_readvariableop_resource:BZ
Lregression__other__person__model_dense_final_biasadd_readvariableop_resource:
identity��>regression__other__person__model/conv0L/BiasAdd/ReadVariableOp�=regression__other__person__model/conv0L/Conv2D/ReadVariableOp�>regression__other__person__model/conv0R/BiasAdd/ReadVariableOp�=regression__other__person__model/conv0R/Conv2D/ReadVariableOp�>regression__other__person__model/conv1L/BiasAdd/ReadVariableOp�=regression__other__person__model/conv1L/Conv2D/ReadVariableOp�>regression__other__person__model/conv1R/BiasAdd/ReadVariableOp�=regression__other__person__model/conv1R/Conv2D/ReadVariableOp�>regression__other__person__model/conv2L/BiasAdd/ReadVariableOp�=regression__other__person__model/conv2L/Conv2D/ReadVariableOp�>regression__other__person__model/conv2R/BiasAdd/ReadVariableOp�=regression__other__person__model/conv2R/Conv2D/ReadVariableOp�>regression__other__person__model/conv3L/BiasAdd/ReadVariableOp�=regression__other__person__model/conv3L/Conv2D/ReadVariableOp�>regression__other__person__model/conv3R/BiasAdd/ReadVariableOp�=regression__other__person__model/conv3R/Conv2D/ReadVariableOp�?regression__other__person__model/dense_0/BiasAdd/ReadVariableOp�>regression__other__person__model/dense_0/MatMul/ReadVariableOp�?regression__other__person__model/dense_1/BiasAdd/ReadVariableOp�>regression__other__person__model/dense_1/MatMul/ReadVariableOp�Cregression__other__person__model/dense_final/BiasAdd/ReadVariableOp�Bregression__other__person__model/dense_final/MatMul/ReadVariableOp{
&regression__other__person__model/ConstConst*
_output_shapes
:*
dtype0*!
valueB"         r
0regression__other__person__model/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&regression__other__person__model/splitSplitVinput_1/regression__other__person__model/Const:output:09regression__other__person__model/split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_split�
.regression__other__person__model/reshape/ShapeShape/regression__other__person__model/split:output:0*
T0*
_output_shapes
:�
<regression__other__person__model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
>regression__other__person__model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
>regression__other__person__model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
6regression__other__person__model/reshape/strided_sliceStridedSlice7regression__other__person__model/reshape/Shape:output:0Eregression__other__person__model/reshape/strided_slice/stack:output:0Gregression__other__person__model/reshape/strided_slice/stack_1:output:0Gregression__other__person__model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8regression__other__person__model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xz
8regression__other__person__model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :z
8regression__other__person__model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
6regression__other__person__model/reshape/Reshape/shapePack?regression__other__person__model/reshape/strided_slice:output:0Aregression__other__person__model/reshape/Reshape/shape/1:output:0Aregression__other__person__model/reshape/Reshape/shape/2:output:0Aregression__other__person__model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
0regression__other__person__model/reshape/ReshapeReshape/regression__other__person__model/split:output:0?regression__other__person__model/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
0regression__other__person__model/reshape_1/ShapeShape/regression__other__person__model/split:output:1*
T0*
_output_shapes
:�
>regression__other__person__model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
@regression__other__person__model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
@regression__other__person__model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
8regression__other__person__model/reshape_1/strided_sliceStridedSlice9regression__other__person__model/reshape_1/Shape:output:0Gregression__other__person__model/reshape_1/strided_slice/stack:output:0Iregression__other__person__model/reshape_1/strided_slice/stack_1:output:0Iregression__other__person__model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:regression__other__person__model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x|
:regression__other__person__model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :|
:regression__other__person__model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
8regression__other__person__model/reshape_1/Reshape/shapePackAregression__other__person__model/reshape_1/strided_slice:output:0Cregression__other__person__model/reshape_1/Reshape/shape/1:output:0Cregression__other__person__model/reshape_1/Reshape/shape/2:output:0Cregression__other__person__model/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
2regression__other__person__model/reshape_1/ReshapeReshape/regression__other__person__model/split:output:1Aregression__other__person__model/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
=regression__other__person__model/conv0L/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv0l_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
.regression__other__person__model/conv0L/Conv2DConv2D9regression__other__person__model/reshape/Reshape:output:0Eregression__other__person__model/conv0L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
�
>regression__other__person__model/conv0L/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv0l_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
/regression__other__person__model/conv0L/BiasAddBiasAdd7regression__other__person__model/conv0L/Conv2D:output:0Fregression__other__person__model/conv0L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v �
,regression__other__person__model/conv0L/ReluRelu8regression__other__person__model/conv0L/BiasAdd:output:0*
T0*/
_output_shapes
:���������v �
1regression__other__person__model/dropout/IdentityIdentity:regression__other__person__model/conv0L/Relu:activations:0*
T0*/
_output_shapes
:���������v �
=regression__other__person__model/conv1L/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv1l_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
.regression__other__person__model/conv1L/Conv2DConv2D:regression__other__person__model/dropout/Identity:output:0Eregression__other__person__model/conv1L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
�
>regression__other__person__model/conv1L/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv1l_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
/regression__other__person__model/conv1L/BiasAddBiasAdd7regression__other__person__model/conv1L/Conv2D:output:0Fregression__other__person__model/conv1L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t �
,regression__other__person__model/conv1L/ReluRelu8regression__other__person__model/conv1L/BiasAdd:output:0*
T0*/
_output_shapes
:���������t �
3regression__other__person__model/dropout/Identity_1Identity:regression__other__person__model/conv1L/Relu:activations:0*
T0*/
_output_shapes
:���������t �
4regression__other__person__model/max_pooling/MaxPoolMaxPool<regression__other__person__model/dropout/Identity_1:output:0*/
_output_shapes
:���������: *
ksize
*
paddingVALID*
strides
�
=regression__other__person__model/conv2L/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv2l_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
.regression__other__person__model/conv2L/Conv2DConv2D=regression__other__person__model/max_pooling/MaxPool:output:0Eregression__other__person__model/conv2L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
�
>regression__other__person__model/conv2L/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv2l_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
/regression__other__person__model/conv2L/BiasAddBiasAdd7regression__other__person__model/conv2L/Conv2D:output:0Fregression__other__person__model/conv2L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@�
,regression__other__person__model/conv2L/ReluRelu8regression__other__person__model/conv2L/BiasAdd:output:0*
T0*/
_output_shapes
:���������8@�
3regression__other__person__model/dropout/Identity_2Identity:regression__other__person__model/conv2L/Relu:activations:0*
T0*/
_output_shapes
:���������8@�
6regression__other__person__model/max_pooling/MaxPool_1MaxPool<regression__other__person__model/dropout/Identity_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
=regression__other__person__model/conv3L/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv3l_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
.regression__other__person__model/conv3L/Conv2DConv2D?regression__other__person__model/max_pooling/MaxPool_1:output:0Eregression__other__person__model/conv3L/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
>regression__other__person__model/conv3L/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv3l_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/regression__other__person__model/conv3L/BiasAddBiasAdd7regression__other__person__model/conv3L/Conv2D:output:0Fregression__other__person__model/conv3L/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
,regression__other__person__model/conv3L/ReluRelu8regression__other__person__model/conv3L/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3regression__other__person__model/dropout/Identity_3Identity:regression__other__person__model/conv3L/Relu:activations:0*
T0*0
_output_shapes
:����������
.regression__other__person__model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
0regression__other__person__model/flatten/ReshapeReshape<regression__other__person__model/dropout/Identity_3:output:07regression__other__person__model/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
=regression__other__person__model/conv0R/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv0r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
.regression__other__person__model/conv0R/Conv2DConv2D;regression__other__person__model/reshape_1/Reshape:output:0Eregression__other__person__model/conv0R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
�
>regression__other__person__model/conv0R/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv0r_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
/regression__other__person__model/conv0R/BiasAddBiasAdd7regression__other__person__model/conv0R/Conv2D:output:0Fregression__other__person__model/conv0R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v �
,regression__other__person__model/conv0R/ReluRelu8regression__other__person__model/conv0R/BiasAdd:output:0*
T0*/
_output_shapes
:���������v �
3regression__other__person__model/dropout/Identity_4Identity:regression__other__person__model/conv0R/Relu:activations:0*
T0*/
_output_shapes
:���������v �
=regression__other__person__model/conv1R/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv1r_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
.regression__other__person__model/conv1R/Conv2DConv2D<regression__other__person__model/dropout/Identity_4:output:0Eregression__other__person__model/conv1R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
�
>regression__other__person__model/conv1R/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv1r_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
/regression__other__person__model/conv1R/BiasAddBiasAdd7regression__other__person__model/conv1R/Conv2D:output:0Fregression__other__person__model/conv1R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t �
,regression__other__person__model/conv1R/ReluRelu8regression__other__person__model/conv1R/BiasAdd:output:0*
T0*/
_output_shapes
:���������t �
3regression__other__person__model/dropout/Identity_5Identity:regression__other__person__model/conv1R/Relu:activations:0*
T0*/
_output_shapes
:���������t �
6regression__other__person__model/max_pooling/MaxPool_2MaxPool<regression__other__person__model/dropout/Identity_5:output:0*/
_output_shapes
:���������: *
ksize
*
paddingVALID*
strides
�
=regression__other__person__model/conv2R/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv2r_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
.regression__other__person__model/conv2R/Conv2DConv2D?regression__other__person__model/max_pooling/MaxPool_2:output:0Eregression__other__person__model/conv2R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
�
>regression__other__person__model/conv2R/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv2r_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
/regression__other__person__model/conv2R/BiasAddBiasAdd7regression__other__person__model/conv2R/Conv2D:output:0Fregression__other__person__model/conv2R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@�
,regression__other__person__model/conv2R/ReluRelu8regression__other__person__model/conv2R/BiasAdd:output:0*
T0*/
_output_shapes
:���������8@�
3regression__other__person__model/dropout/Identity_6Identity:regression__other__person__model/conv2R/Relu:activations:0*
T0*/
_output_shapes
:���������8@�
6regression__other__person__model/max_pooling/MaxPool_3MaxPool<regression__other__person__model/dropout/Identity_6:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
=regression__other__person__model/conv3R/Conv2D/ReadVariableOpReadVariableOpFregression__other__person__model_conv3r_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
.regression__other__person__model/conv3R/Conv2DConv2D?regression__other__person__model/max_pooling/MaxPool_3:output:0Eregression__other__person__model/conv3R/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
>regression__other__person__model/conv3R/BiasAdd/ReadVariableOpReadVariableOpGregression__other__person__model_conv3r_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/regression__other__person__model/conv3R/BiasAddBiasAdd7regression__other__person__model/conv3R/Conv2D:output:0Fregression__other__person__model/conv3R/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
,regression__other__person__model/conv3R/ReluRelu8regression__other__person__model/conv3R/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3regression__other__person__model/dropout/Identity_7Identity:regression__other__person__model/conv3R/Relu:activations:0*
T0*0
_output_shapes
:�����������
0regression__other__person__model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
2regression__other__person__model/flatten/Reshape_1Reshape<regression__other__person__model/dropout/Identity_7:output:09regression__other__person__model/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������z
8regression__other__person__model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
3regression__other__person__model/concatenate/concatConcatV29regression__other__person__model/flatten/Reshape:output:0;regression__other__person__model/flatten/Reshape_1:output:0Aregression__other__person__model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
>regression__other__person__model/dense_0/MatMul/ReadVariableOpReadVariableOpGregression__other__person__model_dense_0_matmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
/regression__other__person__model/dense_0/MatMulMatMul<regression__other__person__model/concatenate/concat:output:0Fregression__other__person__model/dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
?regression__other__person__model/dense_0/BiasAdd/ReadVariableOpReadVariableOpHregression__other__person__model_dense_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0regression__other__person__model/dense_0/BiasAddBiasAdd9regression__other__person__model/dense_0/MatMul:product:0Gregression__other__person__model/dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>regression__other__person__model/dense_0/leaky_re_lu/LeakyRelu	LeakyRelu9regression__other__person__model/dense_0/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%��u=|
:regression__other__person__model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
5regression__other__person__model/concatenate_1/concatConcatV2Lregression__other__person__model/dense_0/leaky_re_lu/LeakyRelu:activations:0/regression__other__person__model/split:output:2Cregression__other__person__model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
3regression__other__person__model/dropout/Identity_8Identity>regression__other__person__model/concatenate_1/concat:output:0*
T0*(
_output_shapes
:�����������
>regression__other__person__model/dense_1/MatMul/ReadVariableOpReadVariableOpGregression__other__person__model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
/regression__other__person__model/dense_1/MatMulMatMul<regression__other__person__model/dropout/Identity_8:output:0Fregression__other__person__model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
?regression__other__person__model/dense_1/BiasAdd/ReadVariableOpReadVariableOpHregression__other__person__model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
0regression__other__person__model/dense_1/BiasAddBiasAdd9regression__other__person__model/dense_1/MatMul:product:0Gregression__other__person__model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
@regression__other__person__model/dense_1/leaky_re_lu_1/LeakyRelu	LeakyRelu9regression__other__person__model/dense_1/BiasAdd:output:0*'
_output_shapes
:���������@*
alpha%��u=|
:regression__other__person__model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
5regression__other__person__model/concatenate_2/concatConcatV2Nregression__other__person__model/dense_1/leaky_re_lu_1/LeakyRelu:activations:0/regression__other__person__model/split:output:2Cregression__other__person__model/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������B�
3regression__other__person__model/dropout/Identity_9Identity>regression__other__person__model/concatenate_2/concat:output:0*
T0*'
_output_shapes
:���������B�
Bregression__other__person__model/dense_final/MatMul/ReadVariableOpReadVariableOpKregression__other__person__model_dense_final_matmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
3regression__other__person__model/dense_final/MatMulMatMul<regression__other__person__model/dropout/Identity_9:output:0Jregression__other__person__model/dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Cregression__other__person__model/dense_final/BiasAdd/ReadVariableOpReadVariableOpLregression__other__person__model_dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
4regression__other__person__model/dense_final/BiasAddBiasAdd=regression__other__person__model/dense_final/MatMul:product:0Kregression__other__person__model/dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4regression__other__person__model/dense_final/SigmoidSigmoid=regression__other__person__model/dense_final/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity8regression__other__person__model/dense_final/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp?^regression__other__person__model/conv0L/BiasAdd/ReadVariableOp>^regression__other__person__model/conv0L/Conv2D/ReadVariableOp?^regression__other__person__model/conv0R/BiasAdd/ReadVariableOp>^regression__other__person__model/conv0R/Conv2D/ReadVariableOp?^regression__other__person__model/conv1L/BiasAdd/ReadVariableOp>^regression__other__person__model/conv1L/Conv2D/ReadVariableOp?^regression__other__person__model/conv1R/BiasAdd/ReadVariableOp>^regression__other__person__model/conv1R/Conv2D/ReadVariableOp?^regression__other__person__model/conv2L/BiasAdd/ReadVariableOp>^regression__other__person__model/conv2L/Conv2D/ReadVariableOp?^regression__other__person__model/conv2R/BiasAdd/ReadVariableOp>^regression__other__person__model/conv2R/Conv2D/ReadVariableOp?^regression__other__person__model/conv3L/BiasAdd/ReadVariableOp>^regression__other__person__model/conv3L/Conv2D/ReadVariableOp?^regression__other__person__model/conv3R/BiasAdd/ReadVariableOp>^regression__other__person__model/conv3R/Conv2D/ReadVariableOp@^regression__other__person__model/dense_0/BiasAdd/ReadVariableOp?^regression__other__person__model/dense_0/MatMul/ReadVariableOp@^regression__other__person__model/dense_1/BiasAdd/ReadVariableOp?^regression__other__person__model/dense_1/MatMul/ReadVariableOpD^regression__other__person__model/dense_final/BiasAdd/ReadVariableOpC^regression__other__person__model/dense_final/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2�
>regression__other__person__model/conv0L/BiasAdd/ReadVariableOp>regression__other__person__model/conv0L/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv0L/Conv2D/ReadVariableOp=regression__other__person__model/conv0L/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv0R/BiasAdd/ReadVariableOp>regression__other__person__model/conv0R/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv0R/Conv2D/ReadVariableOp=regression__other__person__model/conv0R/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv1L/BiasAdd/ReadVariableOp>regression__other__person__model/conv1L/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv1L/Conv2D/ReadVariableOp=regression__other__person__model/conv1L/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv1R/BiasAdd/ReadVariableOp>regression__other__person__model/conv1R/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv1R/Conv2D/ReadVariableOp=regression__other__person__model/conv1R/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv2L/BiasAdd/ReadVariableOp>regression__other__person__model/conv2L/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv2L/Conv2D/ReadVariableOp=regression__other__person__model/conv2L/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv2R/BiasAdd/ReadVariableOp>regression__other__person__model/conv2R/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv2R/Conv2D/ReadVariableOp=regression__other__person__model/conv2R/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv3L/BiasAdd/ReadVariableOp>regression__other__person__model/conv3L/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv3L/Conv2D/ReadVariableOp=regression__other__person__model/conv3L/Conv2D/ReadVariableOp2�
>regression__other__person__model/conv3R/BiasAdd/ReadVariableOp>regression__other__person__model/conv3R/BiasAdd/ReadVariableOp2~
=regression__other__person__model/conv3R/Conv2D/ReadVariableOp=regression__other__person__model/conv3R/Conv2D/ReadVariableOp2�
?regression__other__person__model/dense_0/BiasAdd/ReadVariableOp?regression__other__person__model/dense_0/BiasAdd/ReadVariableOp2�
>regression__other__person__model/dense_0/MatMul/ReadVariableOp>regression__other__person__model/dense_0/MatMul/ReadVariableOp2�
?regression__other__person__model/dense_1/BiasAdd/ReadVariableOp?regression__other__person__model/dense_1/BiasAdd/ReadVariableOp2�
>regression__other__person__model/dense_1/MatMul/ReadVariableOp>regression__other__person__model/dense_1/MatMul/ReadVariableOp2�
Cregression__other__person__model/dense_final/BiasAdd/ReadVariableOpCregression__other__person__model/dense_final/BiasAdd/ReadVariableOp2�
Bregression__other__person__model/dense_final/MatMul/ReadVariableOpBregression__other__person__model/dense_final/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������p
!
_user_specified_name	input_1
ْ
�
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9874
input_1%
conv0l_9730: 
conv0l_9732: %
conv1l_9736:  
conv1l_9738: %
conv2l_9743: @
conv2l_9745:@&
conv3l_9750:@�
conv3l_9752:	�%
conv0r_9757: 
conv0r_9759: %
conv1r_9763:  
conv1r_9765: %
conv2r_9770: @
conv2r_9772:@&
conv3r_9777:@�
conv3r_9779:	� 
dense_0_9786:
�4�
dense_0_9788:	�
dense_1_9794:	�@
dense_1_9796:@"
dense_final_9802:B
dense_final_9804:
identity��conv0L/StatefulPartitionedCall�conv0R/StatefulPartitionedCall�conv1L/StatefulPartitionedCall�conv1R/StatefulPartitionedCall�conv2L/StatefulPartitionedCall�conv2R/StatefulPartitionedCall�conv3L/StatefulPartitionedCall�conv3R/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�#dense_final/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout/StatefulPartitionedCall_1�!dropout/StatefulPartitionedCall_2�!dropout/StatefulPartitionedCall_3�!dropout/StatefulPartitionedCall_4�!dropout/StatefulPartitionedCall_5�!dropout/StatefulPartitionedCall_6�!dropout/StatefulPartitionedCall_7�!dropout/StatefulPartitionedCall_8�!dropout/StatefulPartitionedCall_9�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitVinput_1Const:output:0split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_splitK
reshape/ShapeShapesplit:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapesplit:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������xM
reshape_1/ShapeShapesplit:output:1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapesplit:output:1 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
conv0L/StatefulPartitionedCallStatefulPartitionedCallreshape/Reshape:output:0conv0l_9730conv0l_9732*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0L_layer_call_and_return_conditional_losses_8552�
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv0L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9158�
conv1L/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1l_9736conv1l_9738*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1L_layer_call_and_return_conditional_losses_8582�
!dropout/StatefulPartitionedCall_1StatefulPartitionedCall'conv1L/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9126�
max_pooling/PartitionedCallPartitionedCall*dropout/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2L/StatefulPartitionedCallStatefulPartitionedCall$max_pooling/PartitionedCall:output:0conv2l_9743conv2l_9745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2L_layer_call_and_return_conditional_losses_8612�
!dropout/StatefulPartitionedCall_2StatefulPartitionedCall'conv2L/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9094�
max_pooling/PartitionedCall_1PartitionedCall*dropout/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3L/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_1:output:0conv3l_9750conv3l_9752*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3L_layer_call_and_return_conditional_losses_8642�
!dropout/StatefulPartitionedCall_3StatefulPartitionedCall'conv3L/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9062�
flatten/PartitionedCallPartitionedCall*dropout/StatefulPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660�
conv0R/StatefulPartitionedCallStatefulPartitionedCallreshape_1/Reshape:output:0conv0r_9757conv0r_9759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0R_layer_call_and_return_conditional_losses_8679�
!dropout/StatefulPartitionedCall_4StatefulPartitionedCall'conv0R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_3*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9158�
conv1R/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_4:output:0conv1r_9763conv1r_9765*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1R_layer_call_and_return_conditional_losses_8703�
!dropout/StatefulPartitionedCall_5StatefulPartitionedCall'conv1R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_4*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9126�
max_pooling/PartitionedCall_2PartitionedCall*dropout/StatefulPartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_2:output:0conv2r_9770conv2r_9772*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2R_layer_call_and_return_conditional_losses_8728�
!dropout/StatefulPartitionedCall_6StatefulPartitionedCall'conv2R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_5*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9094�
max_pooling/PartitionedCall_3PartitionedCall*dropout/StatefulPartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_3:output:0conv3r_9777conv3r_9779*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3R_layer_call_and_return_conditional_losses_8753�
!dropout/StatefulPartitionedCall_7StatefulPartitionedCall'conv3R/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9062�
flatten/PartitionedCall_1PartitionedCall*dropout/StatefulPartitionedCall_7:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2 flatten/PartitionedCall:output:0"flatten/PartitionedCall_1:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
dense_0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0dense_0_9786dense_0_9788*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_0_layer_call_and_return_conditional_losses_8780[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2(dense_0/StatefulPartitionedCall:output:0split:output:2"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
!dropout/StatefulPartitionedCall_8StatefulPartitionedCallconcatenate_1/concat:output:0"^dropout/StatefulPartitionedCall_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9024�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_8:output:0dense_1_9794dense_1_9796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_8811[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2(dense_1/StatefulPartitionedCall:output:0split:output:2"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������B�
!dropout/StatefulPartitionedCall_9StatefulPartitionedCallconcatenate_2/concat:output:0"^dropout/StatefulPartitionedCall_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������B* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8992�
#dense_final/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_9:output:0dense_final_9802dense_final_9804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_final_layer_call_and_return_conditional_losses_8842�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0l_9730*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1l_9736*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2l_9743*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3l_9750*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0r_9757*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1r_9763*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2r_9770*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3r_9777*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_9786* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_9794*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_9802*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv0L/StatefulPartitionedCall^conv0R/StatefulPartitionedCall^conv1L/StatefulPartitionedCall^conv1R/StatefulPartitionedCall^conv2L/StatefulPartitionedCall^conv2R/StatefulPartitionedCall^conv3L/StatefulPartitionedCall^conv3R/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^dense_final/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout/StatefulPartitionedCall_1"^dropout/StatefulPartitionedCall_2"^dropout/StatefulPartitionedCall_3"^dropout/StatefulPartitionedCall_4"^dropout/StatefulPartitionedCall_5"^dropout/StatefulPartitionedCall_6"^dropout/StatefulPartitionedCall_7"^dropout/StatefulPartitionedCall_8"^dropout/StatefulPartitionedCall_9Q^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2@
conv0L/StatefulPartitionedCallconv0L/StatefulPartitionedCall2@
conv0R/StatefulPartitionedCallconv0R/StatefulPartitionedCall2@
conv1L/StatefulPartitionedCallconv1L/StatefulPartitionedCall2@
conv1R/StatefulPartitionedCallconv1R/StatefulPartitionedCall2@
conv2L/StatefulPartitionedCallconv2L/StatefulPartitionedCall2@
conv2R/StatefulPartitionedCallconv2R/StatefulPartitionedCall2@
conv3L/StatefulPartitionedCallconv3L/StatefulPartitionedCall2@
conv3R/StatefulPartitionedCallconv3R/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout/StatefulPartitionedCall_1!dropout/StatefulPartitionedCall_12F
!dropout/StatefulPartitionedCall_2!dropout/StatefulPartitionedCall_22F
!dropout/StatefulPartitionedCall_3!dropout/StatefulPartitionedCall_32F
!dropout/StatefulPartitionedCall_4!dropout/StatefulPartitionedCall_42F
!dropout/StatefulPartitionedCall_5!dropout/StatefulPartitionedCall_52F
!dropout/StatefulPartitionedCall_6!dropout/StatefulPartitionedCall_62F
!dropout/StatefulPartitionedCall_7!dropout/StatefulPartitionedCall_72F
!dropout/StatefulPartitionedCall_8!dropout/StatefulPartitionedCall_82F
!dropout/StatefulPartitionedCall_9!dropout/StatefulPartitionedCall_92�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:Q M
(
_output_shapes
:����������p
!
_user_specified_name	input_1
��
�
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9702
input_1%
conv0l_9558: 
conv0l_9560: %
conv1l_9564:  
conv1l_9566: %
conv2l_9571: @
conv2l_9573:@&
conv3l_9578:@�
conv3l_9580:	�%
conv0r_9585: 
conv0r_9587: %
conv1r_9591:  
conv1r_9593: %
conv2r_9598: @
conv2r_9600:@&
conv3r_9605:@�
conv3r_9607:	� 
dense_0_9614:
�4�
dense_0_9616:	�
dense_1_9622:	�@
dense_1_9624:@"
dense_final_9630:B
dense_final_9632:
identity��conv0L/StatefulPartitionedCall�conv0R/StatefulPartitionedCall�conv1L/StatefulPartitionedCall�conv1R/StatefulPartitionedCall�conv2L/StatefulPartitionedCall�conv2R/StatefulPartitionedCall�conv3L/StatefulPartitionedCall�conv3R/StatefulPartitionedCall�dense_0/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�#dense_final/StatefulPartitionedCall�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitVinput_1Const:output:0split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_splitK
reshape/ShapeShapesplit:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapesplit:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������xM
reshape_1/ShapeShapesplit:output:1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapesplit:output:1 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
conv0L/StatefulPartitionedCallStatefulPartitionedCallreshape/Reshape:output:0conv0l_9558conv0l_9560*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0L_layer_call_and_return_conditional_losses_8552�
dropout/PartitionedCallPartitionedCall'conv0L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8563�
conv1L/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1l_9564conv1l_9566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1L_layer_call_and_return_conditional_losses_8582�
dropout/PartitionedCall_1PartitionedCall'conv1L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8592�
max_pooling/PartitionedCallPartitionedCall"dropout/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2L/StatefulPartitionedCallStatefulPartitionedCall$max_pooling/PartitionedCall:output:0conv2l_9571conv2l_9573*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2L_layer_call_and_return_conditional_losses_8612�
dropout/PartitionedCall_2PartitionedCall'conv2L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8622�
max_pooling/PartitionedCall_1PartitionedCall"dropout/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3L/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_1:output:0conv3l_9578conv3l_9580*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3L_layer_call_and_return_conditional_losses_8642�
dropout/PartitionedCall_3PartitionedCall'conv3L/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8652�
flatten/PartitionedCallPartitionedCall"dropout/PartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660�
conv0R/StatefulPartitionedCallStatefulPartitionedCallreshape_1/Reshape:output:0conv0r_9585conv0r_9587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0R_layer_call_and_return_conditional_losses_8679�
dropout/PartitionedCall_4PartitionedCall'conv0R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8563�
conv1R/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_4:output:0conv1r_9591conv1r_9593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv1R_layer_call_and_return_conditional_losses_8703�
dropout/PartitionedCall_5PartitionedCall'conv1R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8592�
max_pooling/PartitionedCall_2PartitionedCall"dropout/PartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv2R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_2:output:0conv2r_9598conv2r_9600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2R_layer_call_and_return_conditional_losses_8728�
dropout/PartitionedCall_6PartitionedCall'conv2R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8622�
max_pooling/PartitionedCall_3PartitionedCall"dropout/PartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
conv3R/StatefulPartitionedCallStatefulPartitionedCall&max_pooling/PartitionedCall_3:output:0conv3r_9605conv3r_9607*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3R_layer_call_and_return_conditional_losses_8753�
dropout/PartitionedCall_7PartitionedCall'conv3R/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8652�
flatten/PartitionedCall_1PartitionedCall"dropout/PartitionedCall_7:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8660Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2 flatten/PartitionedCall:output:0"flatten/PartitionedCall_1:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
dense_0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0dense_0_9614dense_0_9616*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_0_layer_call_and_return_conditional_losses_8780[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2(dense_0/StatefulPartitionedCall:output:0split:output:2"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dropout/PartitionedCall_8PartitionedCallconcatenate_1/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8792�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_8:output:0dense_1_9622dense_1_9624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_8811[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2(dense_1/StatefulPartitionedCall:output:0split:output:2"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������B�
dropout/PartitionedCall_9PartitionedCallconcatenate_2/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������B* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8823�
#dense_final/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_9:output:0dense_final_9630dense_final_9632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_final_layer_call_and_return_conditional_losses_8842�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0l_9558*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1l_9564*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2l_9571*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3l_9578*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv0r_9585*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1r_9591*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2r_9598*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3r_9605*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_0_9614* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_9622*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_9630*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp^conv0L/StatefulPartitionedCall^conv0R/StatefulPartitionedCall^conv1L/StatefulPartitionedCall^conv1R/StatefulPartitionedCall^conv2L/StatefulPartitionedCall^conv2R/StatefulPartitionedCall^conv3L/StatefulPartitionedCall^conv3R/StatefulPartitionedCall ^dense_0/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^dense_final/StatefulPartitionedCallQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2@
conv0L/StatefulPartitionedCallconv0L/StatefulPartitionedCall2@
conv0R/StatefulPartitionedCallconv0R/StatefulPartitionedCall2@
conv1L/StatefulPartitionedCallconv1L/StatefulPartitionedCall2@
conv1R/StatefulPartitionedCallconv1R/StatefulPartitionedCall2@
conv2L/StatefulPartitionedCallconv2L/StatefulPartitionedCall2@
conv2R/StatefulPartitionedCallconv2R/StatefulPartitionedCall2@
conv3L/StatefulPartitionedCallconv3L/StatefulPartitionedCall2@
conv3R/StatefulPartitionedCallconv3R/StatefulPartitionedCall2B
dense_0/StatefulPartitionedCalldense_0/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:Q M
(
_output_shapes
:����������p
!
_user_specified_name	input_1
��
�
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10291

inputs?
%conv0l_conv2d_readvariableop_resource: 4
&conv0l_biasadd_readvariableop_resource: ?
%conv1l_conv2d_readvariableop_resource:  4
&conv1l_biasadd_readvariableop_resource: ?
%conv2l_conv2d_readvariableop_resource: @4
&conv2l_biasadd_readvariableop_resource:@@
%conv3l_conv2d_readvariableop_resource:@�5
&conv3l_biasadd_readvariableop_resource:	�?
%conv0r_conv2d_readvariableop_resource: 4
&conv0r_biasadd_readvariableop_resource: ?
%conv1r_conv2d_readvariableop_resource:  4
&conv1r_biasadd_readvariableop_resource: ?
%conv2r_conv2d_readvariableop_resource: @4
&conv2r_biasadd_readvariableop_resource:@@
%conv3r_conv2d_readvariableop_resource:@�5
&conv3r_biasadd_readvariableop_resource:	�:
&dense_0_matmul_readvariableop_resource:
�4�6
'dense_0_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@<
*dense_final_matmul_readvariableop_resource:B9
+dense_final_biasadd_readvariableop_resource:
identity��conv0L/BiasAdd/ReadVariableOp�conv0L/Conv2D/ReadVariableOp�conv0R/BiasAdd/ReadVariableOp�conv0R/Conv2D/ReadVariableOp�conv1L/BiasAdd/ReadVariableOp�conv1L/Conv2D/ReadVariableOp�conv1R/BiasAdd/ReadVariableOp�conv1R/Conv2D/ReadVariableOp�conv2L/BiasAdd/ReadVariableOp�conv2L/Conv2D/ReadVariableOp�conv2R/BiasAdd/ReadVariableOp�conv2R/Conv2D/ReadVariableOp�conv3L/BiasAdd/ReadVariableOp�conv3L/Conv2D/ReadVariableOp�conv3R/BiasAdd/ReadVariableOp�conv3R/Conv2D/ReadVariableOp�dense_0/BiasAdd/ReadVariableOp�dense_0/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�"dense_final/BiasAdd/ReadVariableOp�!dense_final/MatMul/ReadVariableOp�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitVinputsConst:output:0split/split_dim:output:0*
T0*

Tlen0*O
_output_shapes=
;:����������8:����������8:���������*
	num_splitK
reshape/ShapeShapesplit:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :xY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshapesplit:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������xM
reshape_1/ShapeShapesplit:output:1*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapesplit:output:1 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x�
conv0L/Conv2D/ReadVariableOpReadVariableOp%conv0l_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv0L/Conv2DConv2Dreshape/Reshape:output:0$conv0L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
�
conv0L/BiasAdd/ReadVariableOpReadVariableOp&conv0l_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv0L/BiasAddBiasAddconv0L/Conv2D:output:0%conv0L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v f
conv0L/ReluReluconv0L/BiasAdd:output:0*
T0*/
_output_shapes
:���������v q
dropout/IdentityIdentityconv0L/Relu:activations:0*
T0*/
_output_shapes
:���������v �
conv1L/Conv2D/ReadVariableOpReadVariableOp%conv1l_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv1L/Conv2DConv2Ddropout/Identity:output:0$conv1L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
�
conv1L/BiasAdd/ReadVariableOpReadVariableOp&conv1l_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv1L/BiasAddBiasAddconv1L/Conv2D:output:0%conv1L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t f
conv1L/ReluReluconv1L/BiasAdd:output:0*
T0*/
_output_shapes
:���������t s
dropout/Identity_1Identityconv1L/Relu:activations:0*
T0*/
_output_shapes
:���������t �
max_pooling/MaxPoolMaxPooldropout/Identity_1:output:0*/
_output_shapes
:���������: *
ksize
*
paddingVALID*
strides
�
conv2L/Conv2D/ReadVariableOpReadVariableOp%conv2l_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2L/Conv2DConv2Dmax_pooling/MaxPool:output:0$conv2L/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
�
conv2L/BiasAdd/ReadVariableOpReadVariableOp&conv2l_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2L/BiasAddBiasAddconv2L/Conv2D:output:0%conv2L/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@f
conv2L/ReluReluconv2L/BiasAdd:output:0*
T0*/
_output_shapes
:���������8@s
dropout/Identity_2Identityconv2L/Relu:activations:0*
T0*/
_output_shapes
:���������8@�
max_pooling/MaxPool_1MaxPooldropout/Identity_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv3L/Conv2D/ReadVariableOpReadVariableOp%conv3l_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv3L/Conv2DConv2Dmax_pooling/MaxPool_1:output:0$conv3L/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv3L/BiasAdd/ReadVariableOpReadVariableOp&conv3l_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3L/BiasAddBiasAddconv3L/Conv2D:output:0%conv3L/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv3L/ReluReluconv3L/BiasAdd:output:0*
T0*0
_output_shapes
:����������t
dropout/Identity_3Identityconv3L/Relu:activations:0*
T0*0
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapedropout/Identity_3:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
conv0R/Conv2D/ReadVariableOpReadVariableOp%conv0r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv0R/Conv2DConv2Dreshape_1/Reshape:output:0$conv0R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
�
conv0R/BiasAdd/ReadVariableOpReadVariableOp&conv0r_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv0R/BiasAddBiasAddconv0R/Conv2D:output:0%conv0R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v f
conv0R/ReluReluconv0R/BiasAdd:output:0*
T0*/
_output_shapes
:���������v s
dropout/Identity_4Identityconv0R/Relu:activations:0*
T0*/
_output_shapes
:���������v �
conv1R/Conv2D/ReadVariableOpReadVariableOp%conv1r_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv1R/Conv2DConv2Ddropout/Identity_4:output:0$conv1R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
�
conv1R/BiasAdd/ReadVariableOpReadVariableOp&conv1r_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv1R/BiasAddBiasAddconv1R/Conv2D:output:0%conv1R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t f
conv1R/ReluReluconv1R/BiasAdd:output:0*
T0*/
_output_shapes
:���������t s
dropout/Identity_5Identityconv1R/Relu:activations:0*
T0*/
_output_shapes
:���������t �
max_pooling/MaxPool_2MaxPooldropout/Identity_5:output:0*/
_output_shapes
:���������: *
ksize
*
paddingVALID*
strides
�
conv2R/Conv2D/ReadVariableOpReadVariableOp%conv2r_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2R/Conv2DConv2Dmax_pooling/MaxPool_2:output:0$conv2R/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
�
conv2R/BiasAdd/ReadVariableOpReadVariableOp&conv2r_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2R/BiasAddBiasAddconv2R/Conv2D:output:0%conv2R/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@f
conv2R/ReluReluconv2R/BiasAdd:output:0*
T0*/
_output_shapes
:���������8@s
dropout/Identity_6Identityconv2R/Relu:activations:0*
T0*/
_output_shapes
:���������8@�
max_pooling/MaxPool_3MaxPooldropout/Identity_6:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
conv3R/Conv2D/ReadVariableOpReadVariableOp%conv3r_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv3R/Conv2DConv2Dmax_pooling/MaxPool_3:output:0$conv3R/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv3R/BiasAdd/ReadVariableOpReadVariableOp&conv3r_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3R/BiasAddBiasAddconv3R/Conv2D:output:0%conv3R/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������g
conv3R/ReluReluconv3R/BiasAdd:output:0*
T0*0
_output_shapes
:����������t
dropout/Identity_7Identityconv3R/Relu:activations:0*
T0*0
_output_shapes
:����������`
flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/Reshape_1Reshapedropout/Identity_7:output:0flatten/Const_1:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0flatten/Reshape_1:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������4�
dense_0/MatMul/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
dense_0/MatMulMatMulconcatenate/concat:output:0%dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_0/BiasAdd/ReadVariableOpReadVariableOp'dense_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_0/BiasAddBiasAdddense_0/MatMul:product:0&dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������~
dense_0/leaky_re_lu/LeakyRelu	LeakyReludense_0/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%��u=[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2+dense_0/leaky_re_lu/LeakyRelu:activations:0split:output:2"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������p
dropout/Identity_8Identityconcatenate_1/concat:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_1/MatMulMatMuldropout/Identity_8:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
dense_1/leaky_re_lu_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:���������@*
alpha%��u=[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2-dense_1/leaky_re_lu_1/LeakyRelu:activations:0split:output:2"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������Bo
dropout/Identity_9Identityconcatenate_2/concat:output:0*
T0*'
_output_shapes
:���������B�
!dense_final/MatMul/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
dense_final/MatMulMatMuldropout/Identity_9:output:0)dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_final/BiasAdd/ReadVariableOpReadVariableOp+dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_final/BiasAddBiasAdddense_final/MatMul:product:0*dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
dense_final/SigmoidSigmoiddense_final/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv0l_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv1l_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2l_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3l_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv0r_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv1r_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2r_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3r_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_0_matmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentitydense_final/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv0L/BiasAdd/ReadVariableOp^conv0L/Conv2D/ReadVariableOp^conv0R/BiasAdd/ReadVariableOp^conv0R/Conv2D/ReadVariableOp^conv1L/BiasAdd/ReadVariableOp^conv1L/Conv2D/ReadVariableOp^conv1R/BiasAdd/ReadVariableOp^conv1R/Conv2D/ReadVariableOp^conv2L/BiasAdd/ReadVariableOp^conv2L/Conv2D/ReadVariableOp^conv2R/BiasAdd/ReadVariableOp^conv2R/Conv2D/ReadVariableOp^conv3L/BiasAdd/ReadVariableOp^conv3L/Conv2D/ReadVariableOp^conv3R/BiasAdd/ReadVariableOp^conv3R/Conv2D/ReadVariableOp^dense_0/BiasAdd/ReadVariableOp^dense_0/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^dense_final/BiasAdd/ReadVariableOp"^dense_final/MatMul/ReadVariableOpQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 2>
conv0L/BiasAdd/ReadVariableOpconv0L/BiasAdd/ReadVariableOp2<
conv0L/Conv2D/ReadVariableOpconv0L/Conv2D/ReadVariableOp2>
conv0R/BiasAdd/ReadVariableOpconv0R/BiasAdd/ReadVariableOp2<
conv0R/Conv2D/ReadVariableOpconv0R/Conv2D/ReadVariableOp2>
conv1L/BiasAdd/ReadVariableOpconv1L/BiasAdd/ReadVariableOp2<
conv1L/Conv2D/ReadVariableOpconv1L/Conv2D/ReadVariableOp2>
conv1R/BiasAdd/ReadVariableOpconv1R/BiasAdd/ReadVariableOp2<
conv1R/Conv2D/ReadVariableOpconv1R/Conv2D/ReadVariableOp2>
conv2L/BiasAdd/ReadVariableOpconv2L/BiasAdd/ReadVariableOp2<
conv2L/Conv2D/ReadVariableOpconv2L/Conv2D/ReadVariableOp2>
conv2R/BiasAdd/ReadVariableOpconv2R/BiasAdd/ReadVariableOp2<
conv2R/Conv2D/ReadVariableOpconv2R/Conv2D/ReadVariableOp2>
conv3L/BiasAdd/ReadVariableOpconv3L/BiasAdd/ReadVariableOp2<
conv3L/Conv2D/ReadVariableOpconv3L/Conv2D/ReadVariableOp2>
conv3R/BiasAdd/ReadVariableOpconv3R/BiasAdd/ReadVariableOp2<
conv3R/Conv2D/ReadVariableOpconv3R/Conv2D/ReadVariableOp2@
dense_0/BiasAdd/ReadVariableOpdense_0/BiasAdd/ReadVariableOp2>
dense_0/MatMul/ReadVariableOpdense_0/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"dense_final/BiasAdd/ReadVariableOp"dense_final/BiasAdd/ReadVariableOp2F
!dense_final/MatMul/ReadVariableOp!dense_final/MatMul/ReadVariableOp2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������p
 
_user_specified_nameinputs
�
�
@__inference_conv3R_layer_call_and_return_conditional_losses_8753

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:�����������
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_9094

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������8@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������8@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������8@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������8@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������8@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������8@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8@:W S
/
_output_shapes
:���������8@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_10656s
Yregression__other__person__model_conv2l_kernel_regularizer_square_readvariableop_resource: @
identity��Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv2l_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv2L/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_1_10645s
Yregression__other__person__model_conv1l_kernel_regularizer_square_readvariableop_resource:  
identity��Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv1l_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv1L/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_0_10634s
Yregression__other__person__model_conv0l_kernel_regularizer_square_readvariableop_resource: 
identity��Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv0l_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv0L/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_9_10733m
Zregression__other__person__model_dense_1_kernel_regularizer_square_readvariableop_resource:	�@
identity��Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpZregression__other__person__model_dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityCregression__other__person__model/dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp
�
�
A__inference_conv2L_layer_call_and_return_conditional_losses_11005

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������8@�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2L/kernel/Regularizer/SumSumEregression__other__person__model/conv2L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2L/kernel/Regularizer/mulMulIregression__other__person__model/conv2L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������8@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������: 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_10_10744p
^regression__other__person__model_dense_final_kernel_regularizer_square_readvariableop_resource:B
identity��Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOp^regression__other__person__model_dense_final_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:B*
dtype0�
Fregression__other__person__model/dense_final/kernel/Regularizer/SquareSquare]regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:B�
Eregression__other__person__model/dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
Cregression__other__person__model/dense_final/kernel/Regularizer/SumSumJregression__other__person__model/dense_final/kernel/Regularizer/Square:y:0Nregression__other__person__model/dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Eregression__other__person__model/dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Cregression__other__person__model/dense_final/kernel/Regularizer/mulMulNregression__other__person__model/dense_final/kernel/Regularizer/mul/x:output:0Lregression__other__person__model/dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityGregression__other__person__model/dense_final/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpV^regression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Uregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOpUregression__other__person__model/dense_final/kernel/Regularizer/Square/ReadVariableOp
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8563

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������v c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������v "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������v :W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_8_10722n
Zregression__other__person__model_dense_0_kernel_regularizer_square_readvariableop_resource:
�4�
identity��Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpZregression__other__person__model_dense_0_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityCregression__other__person__model/dense_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp
�
�
@__inference_conv2R_layer_call_and_return_conditional_losses_8728

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������8@�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������8@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������: 
 
_user_specified_nameinputs
�
�
@__inference_conv1R_layer_call_and_return_conditional_losses_8703

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������t �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������t �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������v : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
�
&__inference_conv2L_layer_call_fn_10988

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2L_layer_call_and_return_conditional_losses_8612w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������8@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������: 
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_11187

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@o
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:���������@*
alpha%��u=�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Bregression__other__person__model/dense_1/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
Aregression__other__person__model/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_1/kernel/Regularizer/SumSumFregression__other__person__model/dense_1/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_1/kernel/Regularizer/mulMulJregression__other__person__model/dense_1/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpR^regression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
Qregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_10810

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8592h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������t "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������t :W S
/
_output_shapes
:���������t 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_10689s
Yregression__other__person__model_conv1r_kernel_regularizer_square_readvariableop_resource:  
identity��Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv1r_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv1R/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp
�
C
'__inference_dropout_layer_call_fn_10820

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8563h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������v "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������v :W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8823

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������B[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������B"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������B:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
`
'__inference_dropout_layer_call_fn_10785

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9024p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_7_10711t
Yregression__other__person__model_conv3r_kernel_regularizer_square_readvariableop_resource:@�
identity��Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv3r_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv3R/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp
�	
a
B__inference_dropout_layer_call_and_return_conditional_losses_10867

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������BC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������B*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������Bo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������Bi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������BY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������B"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������B:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_10790

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8652i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
a
B__inference_dropout_layer_call_and_return_conditional_losses_10879

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_conv0L_layer_call_fn_10936

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv0L_layer_call_and_return_conditional_losses_8552w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������v `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������x
 
_user_specified_nameinputs
�
�
A__inference_conv0L_layer_call_and_return_conditional_losses_10953

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������v �
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������v �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������x
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_10850

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_conv3R_layer_call_fn_11118

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv3R_layer_call_and_return_conditional_losses_8753x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_conv1L_layer_call_and_return_conditional_losses_10979

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������t �
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1L/kernel/Regularizer/SumSumEregression__other__person__model/conv1L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1L/kernel/Regularizer/mulMulIregression__other__person__model/conv1L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������t �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������v : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_10915

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������t C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������t *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������t w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������t q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������t a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������t "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������t :W S
/
_output_shapes
:���������t 
 
_user_specified_nameinputs
�
�
A__inference_conv0R_layer_call_and_return_conditional_losses_11057

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������v �
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0R/kernel/Regularizer/SumSumEregression__other__person__model/conv0R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0R/kernel/Regularizer/mulMulIregression__other__person__model/conv0R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������v �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������x
 
_user_specified_nameinputs
�
�
A__inference_conv1R_layer_call_and_return_conditional_losses_11083

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������t X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������t �
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Aregression__other__person__model/conv1R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  �
@regression__other__person__model/conv1R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv1R/kernel/Regularizer/SumSumEregression__other__person__model/conv1R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv1R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv1R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv1R/kernel/Regularizer/mulMulIregression__other__person__model/conv1R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv1R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������t �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������v : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv1R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
�
A__inference_conv3R_layer_call_and_return_conditional_losses_11135

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:�����������
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3R/kernel/Regularizer/SumSumEregression__other__person__model/conv3R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3R/kernel/Regularizer/mulMulIregression__other__person__model/conv3R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_dense_0_layer_call_and_return_conditional_losses_11161

inputs2
matmul_readvariableop_resource:
�4�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������n
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������*
alpha%��u=�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�4�*
dtype0�
Bregression__other__person__model/dense_0/kernel/Regularizer/SquareSquareYregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�4��
Aregression__other__person__model/dense_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?regression__other__person__model/dense_0/kernel/Regularizer/SumSumFregression__other__person__model/dense_0/kernel/Regularizer/Square:y:0Jregression__other__person__model/dense_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
Aregression__other__person__model/dense_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
?regression__other__person__model/dense_0/kernel/Regularizer/mulMulJregression__other__person__model/dense_0/kernel/Regularizer/mul/x:output:0Hregression__other__person__model/dense_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: s
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpR^regression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������4: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
Qregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOpQregression__other__person__model/dense_0/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������4
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8592

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������t c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������t "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������t :W S
/
_output_shapes
:���������t 
 
_user_specified_nameinputs
�
�
?__inference_regression__other__person__model_layer_call_fn_8962
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: @

unknown_12:@%

unknown_13:@�

unknown_14:	�

unknown_15:
�4�

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:B

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *c
f^R\
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_8915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������p: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������p
!
_user_specified_name	input_1
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_10927

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������v C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������v *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������v w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������v q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������v a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������v "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������v :W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
�
@__inference_conv0L_layer_call_and_return_conditional_losses_8552

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������v �
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Aregression__other__person__model/conv0L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv0L/kernel/Regularizer/SumSumEregression__other__person__model/conv0L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv0L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv0L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv0L/kernel/Regularizer/mulMulIregression__other__person__model/conv0L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv0L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������v �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������x: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv0L/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������x
 
_user_specified_nameinputs
�
�
A__inference_conv2R_layer_call_and_return_conditional_losses_11109

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������8@�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Aregression__other__person__model/conv2R/kernel/Regularizer/SquareSquareXregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @�
@regression__other__person__model/conv2R/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv2R/kernel/Regularizer/SumSumEregression__other__person__model/conv2R/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv2R/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv2R/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv2R/kernel/Regularizer/mulMulIregression__other__person__model/conv2R/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv2R/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������8@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpQ^regression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2�
Pregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv2R/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������: 
 
_user_specified_nameinputs
�
`
'__inference_dropout_layer_call_fn_10815

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������t * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9126w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������t `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������t 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������t 
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_10780

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8792a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_max_pooling_layer_call_fn_10760

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_max_pooling_layer_call_and_return_conditional_losses_8500�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_10667t
Yregression__other__person__model_conv3l_kernel_regularizer_square_readvariableop_resource:@�
identity��Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYregression__other__person__model_conv3l_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Aregression__other__person__model/conv3L/kernel/Regularizer/SquareSquareXregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@��
@regression__other__person__model/conv3L/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             �
>regression__other__person__model/conv3L/kernel/Regularizer/SumSumEregression__other__person__model/conv3L/kernel/Regularizer/Square:y:0Iregression__other__person__model/conv3L/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
@regression__other__person__model/conv3L/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
>regression__other__person__model/conv3L/kernel/Regularizer/mulMulIregression__other__person__model/conv3L/kernel/Regularizer/mul/x:output:0Gregression__other__person__model/conv3L/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
IdentityIdentityBregression__other__person__model/conv3L/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpQ^regression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Pregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOpPregression__other__person__model/conv3L/kernel/Regularizer/Square/ReadVariableOp
�
`
'__inference_dropout_layer_call_fn_10825

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������v * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_9158w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������v `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������v 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������v 
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_8660

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_10770

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������B* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8823`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������B"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������B:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������p<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
flatten
	pooling

dropout
conv_0L
conv_1L
conv_2L
conv_3L
conv_0R
conv_1R
conv_2R
conv_3R
dense_0
dense_1
final_dense
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15
(16
)17
*18
+19
,20
-21"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
 8
!9
"10
#11
$12
%13
&14
'15
(16
)17
*18
+19
,20
-21"
trackable_list_wrapper
n
.0
/1
02
13
24
35
46
57
68
79
810"
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
>trace_0
?trace_1
@trace_2
Atrace_32�
?__inference_regression__other__person__model_layer_call_fn_8962
@__inference_regression__other__person__model_layer_call_fn_10046
@__inference_regression__other__person__model_layer_call_fn_10095
?__inference_regression__other__person__model_layer_call_fn_9530�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z>trace_0z?trace_1z@trace_2zAtrace_3
�
Btrace_0
Ctrace_1
Dtrace_2
Etrace_32�
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10291
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10557
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9702
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9874�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zBtrace_0zCtrace_1zDtrace_2zEtrace_3
�B�
__inference__wrapped_model_8491input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
X_random_generator"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

kernel
bias
 __jit_compiled_convolution_op"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

kernel
bias
 f_jit_compiled_convolution_op"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

kernel
bias
 m_jit_compiled_convolution_op"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

kernel
bias
 t_jit_compiled_convolution_op"
_tf_keras_layer
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

 kernel
!bias
 {_jit_compiled_convolution_op"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses

"kernel
#bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

$kernel
%bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

&kernel
'bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�
activation

(kernel
)bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�
activation

*kernel
+bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�m�m�m�m�m�m� m�!m�"m�#m�$m�%m�&m�'m�(m�)m�*m�+m�,m�-m�v�v�v�v�v�v�v�v� v�!v�"v�#v�$v�%v�&v�'v�(v�)v�*v�+v�,v�-v�"
	optimizer
-
�serving_default"
signature_map
H:F 2.regression__other__person__model/conv0L/kernel
::8 2,regression__other__person__model/conv0L/bias
H:F  2.regression__other__person__model/conv1L/kernel
::8 2,regression__other__person__model/conv1L/bias
H:F @2.regression__other__person__model/conv2L/kernel
::8@2,regression__other__person__model/conv2L/bias
I:G@�2.regression__other__person__model/conv3L/kernel
;:9�2,regression__other__person__model/conv3L/bias
H:F 2.regression__other__person__model/conv0R/kernel
::8 2,regression__other__person__model/conv0R/bias
H:F  2.regression__other__person__model/conv1R/kernel
::8 2,regression__other__person__model/conv1R/bias
H:F @2.regression__other__person__model/conv2R/kernel
::8@2,regression__other__person__model/conv2R/bias
I:G@�2.regression__other__person__model/conv3R/kernel
;:9�2,regression__other__person__model/conv3R/bias
C:A
�4�2/regression__other__person__model/dense_0/kernel
<::�2-regression__other__person__model/dense_0/bias
B:@	�@2/regression__other__person__model/dense_1/kernel
;:9@2-regression__other__person__model/dense_1/bias
E:CB23regression__other__person__model/dense_final/kernel
?:=21regression__other__person__model/dense_final/bias
�
�trace_02�
__inference_loss_fn_0_10634�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_10645�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_10656�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_10667�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_10678�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_10689�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_10700�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_10711�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_10722�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_9_10733�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_10_10744�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
?__inference_regression__other__person__model_layer_call_fn_8962input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_regression__other__person__model_layer_call_fn_10046inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_regression__other__person__model_layer_call_fn_10095inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
?__inference_regression__other__person__model_layer_call_fn_9530input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10291inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10557inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9702input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9874input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_10749�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_10755�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_max_pooling_layer_call_fn_10760�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_max_pooling_layer_call_and_return_conditional_losses_10765�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_112�
'__inference_dropout_layer_call_fn_10770
'__inference_dropout_layer_call_fn_10775
'__inference_dropout_layer_call_fn_10780
'__inference_dropout_layer_call_fn_10785
'__inference_dropout_layer_call_fn_10790
'__inference_dropout_layer_call_fn_10795
'__inference_dropout_layer_call_fn_10800
'__inference_dropout_layer_call_fn_10805
'__inference_dropout_layer_call_fn_10810
'__inference_dropout_layer_call_fn_10815
'__inference_dropout_layer_call_fn_10820
'__inference_dropout_layer_call_fn_10825�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11
�

�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_112�
B__inference_dropout_layer_call_and_return_conditional_losses_10830
B__inference_dropout_layer_call_and_return_conditional_losses_10835
B__inference_dropout_layer_call_and_return_conditional_losses_10840
B__inference_dropout_layer_call_and_return_conditional_losses_10845
B__inference_dropout_layer_call_and_return_conditional_losses_10850
B__inference_dropout_layer_call_and_return_conditional_losses_10855
B__inference_dropout_layer_call_and_return_conditional_losses_10867
B__inference_dropout_layer_call_and_return_conditional_losses_10879
B__inference_dropout_layer_call_and_return_conditional_losses_10891
B__inference_dropout_layer_call_and_return_conditional_losses_10903
B__inference_dropout_layer_call_and_return_conditional_losses_10915
B__inference_dropout_layer_call_and_return_conditional_losses_10927�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv0L_layer_call_fn_10936�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv0L_layer_call_and_return_conditional_losses_10953�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv1L_layer_call_fn_10962�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv1L_layer_call_and_return_conditional_losses_10979�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
00"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv2L_layer_call_fn_10988�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv2L_layer_call_and_return_conditional_losses_11005�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
10"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv3L_layer_call_fn_11014�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv3L_layer_call_and_return_conditional_losses_11031�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
'
20"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv0R_layer_call_fn_11040�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv0R_layer_call_and_return_conditional_losses_11057�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
'
30"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv1R_layer_call_fn_11066�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv1R_layer_call_and_return_conditional_losses_11083�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
'
40"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv2R_layer_call_fn_11092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv2R_layer_call_and_return_conditional_losses_11109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
'
50"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv3R_layer_call_fn_11118�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv3R_layer_call_and_return_conditional_losses_11135�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
'
60"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_0_layer_call_fn_11144�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_0_layer_call_and_return_conditional_losses_11161�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
'
70"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_11170�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_11187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
'
80"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_final_layer_call_fn_11196�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_final_layer_call_and_return_conditional_losses_11213�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
"__inference_signature_wrapper_9997input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_10634"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_10645"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_10656"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_10667"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_10678"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_10689"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_10700"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_7_10711"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_8_10722"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_9_10733"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_10_10744"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
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
�B�
'__inference_flatten_layer_call_fn_10749inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_10755inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_max_pooling_layer_call_fn_10760inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_max_pooling_layer_call_and_return_conditional_losses_10765inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
'__inference_dropout_layer_call_fn_10770inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10775inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10780inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10785inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10790inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10795inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10800inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10805inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10810inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10815inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10820inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_10825inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10830inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10835inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10840inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10845inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10850inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10855inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10867inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10879inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10891inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10903inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10915inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_10927inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv0L_layer_call_fn_10936inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv0L_layer_call_and_return_conditional_losses_10953inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv1L_layer_call_fn_10962inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv1L_layer_call_and_return_conditional_losses_10979inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv2L_layer_call_fn_10988inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv2L_layer_call_and_return_conditional_losses_11005inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv3L_layer_call_fn_11014inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv3L_layer_call_and_return_conditional_losses_11031inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv0R_layer_call_fn_11040inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv0R_layer_call_and_return_conditional_losses_11057inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
30"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv1R_layer_call_fn_11066inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv1R_layer_call_and_return_conditional_losses_11083inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv2R_layer_call_fn_11092inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv2R_layer_call_and_return_conditional_losses_11109inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv3R_layer_call_fn_11118inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv3R_layer_call_and_return_conditional_losses_11135inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_0_layer_call_fn_11144inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_0_layer_call_and_return_conditional_losses_11161inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_11170inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_11187inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_final_layer_call_fn_11196inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_final_layer_call_and_return_conditional_losses_11213inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
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
M:K 25Adam/regression__other__person__model/conv0L/kernel/m
?:= 23Adam/regression__other__person__model/conv0L/bias/m
M:K  25Adam/regression__other__person__model/conv1L/kernel/m
?:= 23Adam/regression__other__person__model/conv1L/bias/m
M:K @25Adam/regression__other__person__model/conv2L/kernel/m
?:=@23Adam/regression__other__person__model/conv2L/bias/m
N:L@�25Adam/regression__other__person__model/conv3L/kernel/m
@:>�23Adam/regression__other__person__model/conv3L/bias/m
M:K 25Adam/regression__other__person__model/conv0R/kernel/m
?:= 23Adam/regression__other__person__model/conv0R/bias/m
M:K  25Adam/regression__other__person__model/conv1R/kernel/m
?:= 23Adam/regression__other__person__model/conv1R/bias/m
M:K @25Adam/regression__other__person__model/conv2R/kernel/m
?:=@23Adam/regression__other__person__model/conv2R/bias/m
N:L@�25Adam/regression__other__person__model/conv3R/kernel/m
@:>�23Adam/regression__other__person__model/conv3R/bias/m
H:F
�4�26Adam/regression__other__person__model/dense_0/kernel/m
A:?�24Adam/regression__other__person__model/dense_0/bias/m
G:E	�@26Adam/regression__other__person__model/dense_1/kernel/m
@:>@24Adam/regression__other__person__model/dense_1/bias/m
J:HB2:Adam/regression__other__person__model/dense_final/kernel/m
D:B28Adam/regression__other__person__model/dense_final/bias/m
M:K 25Adam/regression__other__person__model/conv0L/kernel/v
?:= 23Adam/regression__other__person__model/conv0L/bias/v
M:K  25Adam/regression__other__person__model/conv1L/kernel/v
?:= 23Adam/regression__other__person__model/conv1L/bias/v
M:K @25Adam/regression__other__person__model/conv2L/kernel/v
?:=@23Adam/regression__other__person__model/conv2L/bias/v
N:L@�25Adam/regression__other__person__model/conv3L/kernel/v
@:>�23Adam/regression__other__person__model/conv3L/bias/v
M:K 25Adam/regression__other__person__model/conv0R/kernel/v
?:= 23Adam/regression__other__person__model/conv0R/bias/v
M:K  25Adam/regression__other__person__model/conv1R/kernel/v
?:= 23Adam/regression__other__person__model/conv1R/bias/v
M:K @25Adam/regression__other__person__model/conv2R/kernel/v
?:=@23Adam/regression__other__person__model/conv2R/bias/v
N:L@�25Adam/regression__other__person__model/conv3R/kernel/v
@:>�23Adam/regression__other__person__model/conv3R/bias/v
H:F
�4�26Adam/regression__other__person__model/dense_0/kernel/v
A:?�24Adam/regression__other__person__model/dense_0/bias/v
G:E	�@26Adam/regression__other__person__model/dense_1/kernel/v
@:>@24Adam/regression__other__person__model/dense_1/bias/v
J:HB2:Adam/regression__other__person__model/dense_final/kernel/v
D:B28Adam/regression__other__person__model/dense_final/bias/v�
__inference__wrapped_model_8491� !"#$%&'()*+,-1�.
'�$
"�
input_1����������p
� "3�0
.
output_1"�
output_1����������
A__inference_conv0L_layer_call_and_return_conditional_losses_10953l7�4
-�*
(�%
inputs���������x
� "-�*
#� 
0���������v 
� �
&__inference_conv0L_layer_call_fn_10936_7�4
-�*
(�%
inputs���������x
� " ����������v �
A__inference_conv0R_layer_call_and_return_conditional_losses_11057l !7�4
-�*
(�%
inputs���������x
� "-�*
#� 
0���������v 
� �
&__inference_conv0R_layer_call_fn_11040_ !7�4
-�*
(�%
inputs���������x
� " ����������v �
A__inference_conv1L_layer_call_and_return_conditional_losses_10979l7�4
-�*
(�%
inputs���������v 
� "-�*
#� 
0���������t 
� �
&__inference_conv1L_layer_call_fn_10962_7�4
-�*
(�%
inputs���������v 
� " ����������t �
A__inference_conv1R_layer_call_and_return_conditional_losses_11083l"#7�4
-�*
(�%
inputs���������v 
� "-�*
#� 
0���������t 
� �
&__inference_conv1R_layer_call_fn_11066_"#7�4
-�*
(�%
inputs���������v 
� " ����������t �
A__inference_conv2L_layer_call_and_return_conditional_losses_11005l7�4
-�*
(�%
inputs���������: 
� "-�*
#� 
0���������8@
� �
&__inference_conv2L_layer_call_fn_10988_7�4
-�*
(�%
inputs���������: 
� " ����������8@�
A__inference_conv2R_layer_call_and_return_conditional_losses_11109l$%7�4
-�*
(�%
inputs���������: 
� "-�*
#� 
0���������8@
� �
&__inference_conv2R_layer_call_fn_11092_$%7�4
-�*
(�%
inputs���������: 
� " ����������8@�
A__inference_conv3L_layer_call_and_return_conditional_losses_11031m7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
&__inference_conv3L_layer_call_fn_11014`7�4
-�*
(�%
inputs���������@
� "!������������
A__inference_conv3R_layer_call_and_return_conditional_losses_11135m&'7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
&__inference_conv3R_layer_call_fn_11118`&'7�4
-�*
(�%
inputs���������@
� "!������������
B__inference_dense_0_layer_call_and_return_conditional_losses_11161^()0�-
&�#
!�
inputs����������4
� "&�#
�
0����������
� |
'__inference_dense_0_layer_call_fn_11144Q()0�-
&�#
!�
inputs����������4
� "������������
B__inference_dense_1_layer_call_and_return_conditional_losses_11187]*+0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� {
'__inference_dense_1_layer_call_fn_11170P*+0�-
&�#
!�
inputs����������
� "����������@�
F__inference_dense_final_layer_call_and_return_conditional_losses_11213\,-/�,
%�"
 �
inputs���������B
� "%�"
�
0���������
� ~
+__inference_dense_final_layer_call_fn_11196O,-/�,
%�"
 �
inputs���������B
� "�����������
B__inference_dropout_layer_call_and_return_conditional_losses_10830l;�8
1�.
(�%
inputs���������v 
p 
� "-�*
#� 
0���������v 
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10835l;�8
1�.
(�%
inputs���������t 
p 
� "-�*
#� 
0���������t 
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10840l;�8
1�.
(�%
inputs���������8@
p 
� "-�*
#� 
0���������8@
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10845n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10850^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10855\3�0
)�&
 �
inputs���������B
p 
� "%�"
�
0���������B
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10867\3�0
)�&
 �
inputs���������B
p
� "%�"
�
0���������B
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10879^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10891n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10903l;�8
1�.
(�%
inputs���������8@
p
� "-�*
#� 
0���������8@
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10915l;�8
1�.
(�%
inputs���������t 
p
� "-�*
#� 
0���������t 
� �
B__inference_dropout_layer_call_and_return_conditional_losses_10927l;�8
1�.
(�%
inputs���������v 
p
� "-�*
#� 
0���������v 
� z
'__inference_dropout_layer_call_fn_10770O3�0
)�&
 �
inputs���������B
p 
� "����������Bz
'__inference_dropout_layer_call_fn_10775O3�0
)�&
 �
inputs���������B
p
� "����������B|
'__inference_dropout_layer_call_fn_10780Q4�1
*�'
!�
inputs����������
p 
� "�����������|
'__inference_dropout_layer_call_fn_10785Q4�1
*�'
!�
inputs����������
p
� "������������
'__inference_dropout_layer_call_fn_10790a<�9
2�/
)�&
inputs����������
p 
� "!������������
'__inference_dropout_layer_call_fn_10795a<�9
2�/
)�&
inputs����������
p
� "!������������
'__inference_dropout_layer_call_fn_10800_;�8
1�.
(�%
inputs���������8@
p 
� " ����������8@�
'__inference_dropout_layer_call_fn_10805_;�8
1�.
(�%
inputs���������8@
p
� " ����������8@�
'__inference_dropout_layer_call_fn_10810_;�8
1�.
(�%
inputs���������t 
p 
� " ����������t �
'__inference_dropout_layer_call_fn_10815_;�8
1�.
(�%
inputs���������t 
p
� " ����������t �
'__inference_dropout_layer_call_fn_10820_;�8
1�.
(�%
inputs���������v 
p 
� " ����������v �
'__inference_dropout_layer_call_fn_10825_;�8
1�.
(�%
inputs���������v 
p
� " ����������v �
B__inference_flatten_layer_call_and_return_conditional_losses_10755b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
'__inference_flatten_layer_call_fn_10749U8�5
.�+
)�&
inputs����������
� "�����������:
__inference_loss_fn_0_10634�

� 
� "� ;
__inference_loss_fn_10_10744,�

� 
� "� :
__inference_loss_fn_1_10645�

� 
� "� :
__inference_loss_fn_2_10656�

� 
� "� :
__inference_loss_fn_3_10667�

� 
� "� :
__inference_loss_fn_4_10678 �

� 
� "� :
__inference_loss_fn_5_10689"�

� 
� "� :
__inference_loss_fn_6_10700$�

� 
� "� :
__inference_loss_fn_7_10711&�

� 
� "� :
__inference_loss_fn_8_10722(�

� 
� "� :
__inference_loss_fn_9_10733*�

� 
� "� �
F__inference_max_pooling_layer_call_and_return_conditional_losses_10765�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
+__inference_max_pooling_layer_call_fn_10760�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10291u !"#$%&'()*+,-4�1
*�'
!�
inputs����������p
p 
� "%�"
�
0���������
� �
[__inference_regression__other__person__model_layer_call_and_return_conditional_losses_10557u !"#$%&'()*+,-4�1
*�'
!�
inputs����������p
p
� "%�"
�
0���������
� �
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9702v !"#$%&'()*+,-5�2
+�(
"�
input_1����������p
p 
� "%�"
�
0���������
� �
Z__inference_regression__other__person__model_layer_call_and_return_conditional_losses_9874v !"#$%&'()*+,-5�2
+�(
"�
input_1����������p
p
� "%�"
�
0���������
� �
@__inference_regression__other__person__model_layer_call_fn_10046h !"#$%&'()*+,-4�1
*�'
!�
inputs����������p
p 
� "�����������
@__inference_regression__other__person__model_layer_call_fn_10095h !"#$%&'()*+,-4�1
*�'
!�
inputs����������p
p
� "�����������
?__inference_regression__other__person__model_layer_call_fn_8962i !"#$%&'()*+,-5�2
+�(
"�
input_1����������p
p 
� "�����������
?__inference_regression__other__person__model_layer_call_fn_9530i !"#$%&'()*+,-5�2
+�(
"�
input_1����������p
p
� "�����������
"__inference_signature_wrapper_9997� !"#$%&'()*+,-<�9
� 
2�/
-
input_1"�
input_1����������p"3�0
.
output_1"�
output_1���������