	??5Φk`@??5Φk`@!??5Φk`@	&?R?Zl??&?R?Zl??!&?R?Zl??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??5Φk`@???;?^@1eȱ??@A!u;?ʣ?I??~?N@Y.c}???*	 ?rh?%T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK??q?ߕ?!a0)κ?:@)??"???1??????5@:Preprocessing2U
Iterator::Model::ParallelMapV27qr?CQ??!g1???3@)7qr?CQ??1g1???3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\?-??e??!??N?Ȑ=@)VfJ?o	??1@8?o3@:Preprocessing2F
Iterator::Modelz?):?˟?!??1?CC@)??n?????1>?????2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceTƿϸ??!?$?T?C$@)Tƿϸ??1?$?T?C$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?G???\??!-??/?N@)?5?;N?q?1q??@q?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?R?Gn?!??N?kX@)?R?Gn?1??N?kX@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&?R?Zl??I?z??;X@Q?K敏@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???;?^@???;?^@!???;?^@      ??!       "	eȱ??@eȱ??@!eȱ??@*      ??!       2	!u;?ʣ?!u;?ʣ?!!u;?ʣ?:	??~?N@??~?N@!??~?N@B      ??!       J	.c}???.c}???!.c}???R      ??!       Z	.c}???.c}???!.c}???b      ??!       JGPUY&?R?Zl??b q?z??;X@y?K敏@