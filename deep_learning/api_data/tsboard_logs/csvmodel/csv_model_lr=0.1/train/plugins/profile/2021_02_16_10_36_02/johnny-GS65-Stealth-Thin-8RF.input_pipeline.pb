	A??h:?N@A??h:?N@!A??h:?N@	???MZ?????MZ??!???MZ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6A??h:?N@?D??ӴL@1?x` ??ARb??v??I|??%j@Yd?=	lν?*	V-?S@2F
Iterator::Model ??q???!@?_<?yE@)??T????1x?t?w?5@:Preprocessing2U
Iterator::Model::ParallelMapV2S?h?w??!	?J??a5@)S?h?w??1	?J??a5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE?4~ᕔ?!???B`29@){?\?&??1??h?0?4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?H?5??!8?k}?7@)?[??.???1?F????*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen2??n??!v??TC$@)n2??n??1v??TC$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???N??!?3??o?L@)??i? ?s?1?5??^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ??	?yk?!$??(??@)J??	?yk?1$??(??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps?V{???!?;???9@)??P?l]?1F??"5@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???MZ??Ib?.cƑX@QwƓ???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D??ӴL@?D??ӴL@!?D??ӴL@      ??!       "	?x` ???x` ??!?x` ??*      ??!       2	Rb??v??Rb??v??!Rb??v??:	|??%j@|??%j@!|??%j@B      ??!       J	d?=	lν?d?=	lν?!d?=	lν?R      ??!       Z	d?=	lν?d?=	lν?!d?=	lν?b      ??!       JGPUY???MZ??b qb?.cƑX@ywƓ???