	w??3a@w??3a@!w??3a@	?}ж{????}ж{???!?}ж{???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6w??3a@?z?|"`@1?J?4?@A:vP????I??>???@Y????C??*	??(\??U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?8'0???!??cnC9@)?w~Q????1?W5/85@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap
H?`???!??{{?=@)???zݒ?1??H5@:Preprocessing2F
Iterator::Model?Z??Ρ?!?h ,??C@)??u?ݑ?1????3@:Preprocessing2U
Iterator::Model::ParallelMapV24??𽿑?!?1U?3@)4??𽿑?1?1U?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?????!v?Eet!@)?????1v?Eet!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV?P?????!
???N@)~?[?~lr?1g}$L?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorh??n?l?!C0??,@)h??n?l?1C0??,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?}ж{???I??[?YFX@Qު?E-?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?z?|"`@?z?|"`@!?z?|"`@      ??!       "	?J?4?@?J?4?@!?J?4?@*      ??!       2	:vP????:vP????!:vP????:	??>???@??>???@!??>???@B      ??!       J	????C??????C??!????C??R      ??!       Z	????C??????C??!????C??b      ??!       JGPUY?}ж{???b q??[?YFX@yު?E-?@