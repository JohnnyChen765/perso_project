	%?YI΅@%?YI΅@!%?YI΅@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-%?YI΅@[?? ?1p@1PU???z@A>>!;oc??I?2???Y!@*	?S㥛?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?l\???!?]J?W@)? ?	???1<V#rO?U@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip ?I????!ͨ?}?@)????,A??1GҤQ?@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle H1@?	??!???X?	@)H1@?	??1???X?	@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?鲘?|??!H???F?@)?鲘?|??1H???F?@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismKt?Y?b??!??(|?@)??y0H??1??t???@:Preprocessing2F
Iterator::Model??J?RН?!'?!Z??@)????qnc?1M?? ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI9.{?0C@Q?ф??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[?? ?1p@[?? ?1p@![?? ?1p@      ??!       "	PU???z@PU???z@!PU???z@*      ??!       2	>>!;oc??>>!;oc??!>>!;oc??:	?2???Y!@?2???Y!@!?2???Y!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q9.{?0C@y?ф??N@