	???(_?`@???(_?`@!???(_?`@	???^?b?????^?b??!???^?b??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???(_?`@??q??
_@1+3???D@A??B˺??I|??@Yƅ!Y???*=
ףpMX@)      =2U
Iterator::Model::ParallelMapV2QN??????!?$??
D@)QN??????1?$??
D@:Preprocessing2F
Iterator::Model&7??5??!?w?\?VL@)l??7???1Φ?	??0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???[???!
?H?[?3@)f-????1?Yy ?"0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapR%?S;??!?T]h?Q3@)?:??????1?p>???(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?uʣ{?!#q??E?@)?uʣ{?1#q??E?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?y?Տ??!?+?,?E@)dY0?Gq?1???&@\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoram???l?!??{??3@)am???l?1??{??3@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???^?b??I.xl?WX@Q[???5?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??q??
_@??q??
_@!??q??
_@      ??!       "	+3???D@+3???D@!+3???D@*      ??!       2	??B˺????B˺??!??B˺??:	|??@|??@!|??@B      ??!       J	ƅ!Y???ƅ!Y???!ƅ!Y???R      ??!       Z	ƅ!Y???ƅ!Y???!ƅ!Y???b      ??!       JGPUY???^?b??b q.xl?WX@y[???5?@