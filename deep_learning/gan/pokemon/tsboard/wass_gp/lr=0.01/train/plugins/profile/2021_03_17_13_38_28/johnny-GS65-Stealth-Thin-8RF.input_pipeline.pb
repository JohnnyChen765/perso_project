	?R?o???@?R?o???@!?R?o???@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?R?o???@ܠ?[;Hn@1?$?*?y@A??a???I??\I@*	F????z@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2CX?%????!?G?V@)؛?????1ę???U@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??x??M??!߅-T,@)??x??M??1߅-T,@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle ?2??3??!?????I@)?2??3??1?????I@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip T㥛? ??!{S?$&@)ѓ2????1	?XQR@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismz?????!?$t?@)?<?r؍?1?=??Q?@:Preprocessing2F
Iterator::Model?R??%???! ?Ǖ?g @)r?Md?g?1?\?`???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 36.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?7???B@Q~??_gO@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ܠ?[;Hn@ܠ?[;Hn@!ܠ?[;Hn@      ??!       "	?$?*?y@?$?*?y@!?$?*?y@*      ??!       2	??a?????a???!??a???:	??\I@??\I@!??\I@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?7???B@y~??_gO@