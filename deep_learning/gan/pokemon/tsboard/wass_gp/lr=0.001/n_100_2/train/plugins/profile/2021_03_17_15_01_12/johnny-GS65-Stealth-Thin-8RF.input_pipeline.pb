	Ǻ???z?@Ǻ???z?@!Ǻ???z?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Ǻ???z?@6?!??"@1$???9?@A??_YiR??I??I`;!@*	?Q??8?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2B?v????!?&@?+X@)?R\U????1?!3αV@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip ?u?????!?属?@)R%?S;??1?$??m@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle ?-?????!?/?Ь?@)?-?????1?/?Ь?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch}!??????!???p???)}!??????1???p???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-?R??!V?k?7F@)WBwI???1???R???:Preprocessing2F
Iterator::ModelC?ʠ????!?'?7??
@)Ƣ??dpd?1<{?#???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@???-{@Q^??&$X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6?!??"@6?!??"@!6?!??"@      ??!       "	$???9?@$???9?@!$???9?@*      ??!       2	??_YiR????_YiR??!??_YiR??:	??I`;!@??I`;!@!??I`;!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@???-{@y^??&$X@