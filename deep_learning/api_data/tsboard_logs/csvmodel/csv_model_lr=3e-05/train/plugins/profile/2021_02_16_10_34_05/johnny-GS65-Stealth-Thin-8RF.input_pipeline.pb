	?q???O@?q???O@!?q???O@	d??#????d??#????!d??#????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?q???O@`??fM@1;??u??Af?B,c??I???Z?@Y!=E7??*	;?O???T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??)r???!??$?<@)??V????1;|C/r58@:Preprocessing2U
Iterator::Model::ParallelMapV2???f???!??1??7@)???f???1??1??7@:Preprocessing2F
Iterator::Model?:pΈ??!_^r??E@)[^??6S??19?q?D4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?+I?????!?Aw?c?#@)?+I?????1?Aw?c?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???QI???!䶯??o3@)c'??>??1,??? #@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???o?4??!????PQL@)??ŉ?vt?1?!!r?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??9?n?!MK???@)??9?n?1MK???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap>x?҆Ò?!t?Χ`?5@)XuV?1a?1?$??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9e??#????I?=h?ՖX@Q,????^??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`??fM@`??fM@!`??fM@      ??!       "	;??u??;??u??!;??u??*      ??!       2	f?B,c??f?B,c??!f?B,c??:	???Z?@???Z?@!???Z?@B      ??!       J	!=E7??!=E7??!!=E7??R      ??!       Z	!=E7??!=E7??!!=E7??b      ??!       JGPUYe??#????b q?=h?ՖX@y,????^??