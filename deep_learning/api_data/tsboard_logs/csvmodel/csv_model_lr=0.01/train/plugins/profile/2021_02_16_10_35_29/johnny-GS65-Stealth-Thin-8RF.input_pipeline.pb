	? ???M@? ???M@!? ???M@	?x?A????x?A???!?x?A???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6? ???M@??????K@1?#?????AS?h?w??I?WY??@Yi??????*	y?&1T@2U
Iterator::Model::ParallelMapV2ׄ?Ơ??!?fӃ6@)ׄ?Ơ??1?fӃ6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatJ?o	????!}??H?9@)A*ŎƑ?1??!L??5@:Preprocessing2F
Iterator::Model?в???!zQ???D@)m????1?l;YC?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????`???!??yJ?w9@)$??(?[??1@???(0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??W?~?!????"@)??W?~?1????"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?q??????!???i6M@)L5??r?1X?<7??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorrQ-"??k?!?:/?W?@)rQ-"??k?1?:/?W?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn??fc%??!?	??1;@)????(@T?1.?s"W???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?x?A???I'i?6-?X@Q??????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????K@??????K@!??????K@      ??!       "	?#??????#?????!?#?????*      ??!       2	S?h?w??S?h?w??!S?h?w??:	?WY??@?WY??@!?WY??@B      ??!       J	i??????i??????!i??????R      ??!       Z	i??????i??????!i??????b      ??!       JGPUY?x?A???b q'i?6-?X@y??????