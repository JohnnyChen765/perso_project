	???Y_?@???Y_?@!???Y_?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???Y_?@?e?I)?o@1?[?Wz@AS ??蝢?I\??F@*	???Sރ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?8EGry??!?H?0x?V@)???Y????1?[???UT@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip ?T4??ή?!?g/F??"@)Ֆ:?????1?????@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismgs?69??!??g6W!@)?$\?#???1k?????@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchF?-t%??!_ݠ???	@)F?-t%??1_ݠ???	@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle ?{?&??!?g?Ɓ?@)?{?&??1?g?Ɓ?@:Preprocessing2F
Iterator::Modelm?kA???!j??z>d"@)e????`k?1?mbF???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noISi?/C@Q㬖C??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?e?I)?o@?e?I)?o@!?e?I)?o@      ??!       "	?[?Wz@?[?Wz@!?[?Wz@*      ??!       2	S ??蝢?S ??蝢?!S ??蝢?:	\??F@\??F@!\??F@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qSi?/C@y㬖C??N@