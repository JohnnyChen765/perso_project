	??!?g{@??!?g{@!??!?g{@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??!?g{@?)?TP?@1/3l?uqz@A?V??y??I?9@0GO@*	?E????t@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2q?0'h???!????V@)??s?????16oE?b?T@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle u><K???!?3"?@)u><K???1?3"?@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip {-??1??!??q?"@)9(a????1U???@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch[??	m??!?F?z??@)[??	m??1?F?z??@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?D?$]3??!?i???@)??e1????1???DXP@:Preprocessing2F
Iterator::ModelJ$??(???!??'Hw0 @)???9]c?1p?|?-g??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?Q? ?@Qr????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)?TP?@?)?TP?@!?)?TP?@      ??!       "	/3l?uqz@/3l?uqz@!/3l?uqz@*      ??!       2	?V??y???V??y??!?V??y??:	?9@0GO@?9@0GO@!?9@0GO@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Q? ?@yr????X@