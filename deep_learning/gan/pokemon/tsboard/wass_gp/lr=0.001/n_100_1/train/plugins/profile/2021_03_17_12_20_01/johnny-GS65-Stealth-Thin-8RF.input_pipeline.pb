	?3w|J?@?3w|J?@!?3w|J?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?3w|J?@H?}8?n@1E)!X?z@Ap??;???I??B?if @*	??ʡ?}@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2x???Ĭ??!?M'?mW@)/\sG???1z?Ô=U@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip F
e??k??!?Ԩ!??!@)Ωd ????1?]Ѷ?@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle ?je?/???!???q_N@)?je?/???1???q_N@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?*?MF???!Ap?`?@)?*?MF???1Ap?`?@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?sD?K???!+A???@)6??
(??1~?w @:Preprocessing2F
Iterator::Model?@ CǞ?!?!??%@)?g^??h?1????_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 36.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI ?R?B@Q?????O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?}8?n@H?}8?n@!H?}8?n@      ??!       "	E)!X?z@E)!X?z@!E)!X?z@*      ??!       2	p??;???p??;???!p??;???:	??B?if @??B?if @!??B?if @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?R?B@y?????O@