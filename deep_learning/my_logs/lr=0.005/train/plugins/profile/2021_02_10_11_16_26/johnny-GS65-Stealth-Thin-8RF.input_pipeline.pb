	ٕ???a@ٕ???a@!ٕ???a@	??????????!?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ٕ???a@??IӠ?`@18h?>?@A@?P?%???I??*??O@Y?,^,??*	y?&1?R@2U
Iterator::Model::ParallelMapV2?r?]????!Pw????4@)?r?]????1Pw????4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??0??B??!&?vM=@)xak?????1??r???4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV?j-̒?!?`~?o?8@)?g@?5??1n*?4@:Preprocessing2F
Iterator::Modelt?!??!:???e}D@)c?: ⮎?1%???14@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceG????y?!?PW!@)G????y?1?PW!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??\k??!?Y??M@)C?O?}:n?1XN??8?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor'?_?i?!%;???@)'?_?i?1%;???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?????I??&'?PX@Q?7??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??IӠ?`@??IӠ?`@!??IӠ?`@      ??!       "	8h?>?@8h?>?@!8h?>?@*      ??!       2	@?P?%???@?P?%???!@?P?%???:	??*??O@??*??O@!??*??O@B      ??!       J	?,^,???,^,??!?,^,??R      ??!       Z	?,^,???,^,??!?,^,??b      ??!       JGPUY?????b q??&'?PX@y?7??@