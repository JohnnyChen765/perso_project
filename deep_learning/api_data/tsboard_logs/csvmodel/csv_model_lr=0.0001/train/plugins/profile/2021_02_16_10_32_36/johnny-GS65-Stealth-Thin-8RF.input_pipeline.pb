	?! 8??K@?! 8??K@!?! 8??K@	->??X??->??X??!->??X??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?! 8??K@?ٮ?7J@1?//?>:??A?\S ????I????@YQ?+????*	??ʡT@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?乾??!?n???>@)?Ϝ?)ǔ?1??9D?U9@:Preprocessing2F
Iterator::Model?$??,??!??T??C@)>??????1?xD[5@:Preprocessing2U
Iterator::Model::ParallelMapV20??9\???!P???2@)0??9\???1P???2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????>??!{'h?~?6@)?}?ƃ-??1Qd???
+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicefl?f?|?!????t!@)fl?f?|?1????t!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip^??I?Ԩ?!o?p?FN@)y"??ps?1????-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?St$??p?!????)?@)?St$??p?1????)?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?5?????!?j?2?=8@)??c?M*Z?1?1ԧ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9.>??X??I?T.?GyX@Q?
oR~(??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ٮ?7J@?ٮ?7J@!?ٮ?7J@      ??!       "	?//?>:???//?>:??!?//?>:??*      ??!       2	?\S ?????\S ????!?\S ????:	????@????@!????@B      ??!       J	Q?+????Q?+????!Q?+????R      ??!       Z	Q?+????Q?+????!Q?+????b      ??!       JGPUY.>??X??b q?T.?GyX@y?
oR~(??