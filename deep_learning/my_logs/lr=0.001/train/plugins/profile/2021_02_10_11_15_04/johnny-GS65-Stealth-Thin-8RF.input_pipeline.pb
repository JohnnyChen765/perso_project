	進l??家@進l??家@!進l??家@	:??V)R?:??V)R?!:??V)R?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6進l??家@?-$`?`@1鰍^[@A??鴽鬢??I椿)?玅@Y??.城Y?*	?&1挑P@2U
Iterator::Model::ParallelMapV29設????!_絯rSd:@)9設????1_絯rSd:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat々?????!v?Q???=@)漙?-;???1G?5?7@:Preprocessing2F
Iterator::Model!?> 和??!I???<E@)姲翻2o??13?.回0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice鐉????!???n?Q'@)鐉????1???n?Q'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap軋?炄???!@鎂)?5@)a?$?茿?1?U?V? $@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??k*??!徬r?&馥@)?放浢bp?1葰G#x?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoreQ閆?o?!?!幌@)eQ閆?o?1?!幌@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9:??V)R?Iok盻5eX@Qg3??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?-$`?`@?-$`?`@!?-$`?`@      ??!       "	鰍^[@鰍^[@!鰍^[@*      ??!       2	??鴽鬢????鴽鬢??!??鴽鬢??:	椿)?玅@椿)?玅@!椿)?玅@B      ??!       J	??.城Y???.城Y?!??.城Y?R      ??!       Z	??.城Y???.城Y?!??.城Y?b      ??!       JGPUY:??V)R?b qok盻5eX@yg3??V@