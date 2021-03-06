?	ٕ???a@ٕ???a@!ٕ???a@	??????????!?????"w
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
	??IӠ?`@??IӠ?`@!??IӠ?`@      ??!       "	8h?>?@8h?>?@!8h?>?@*      ??!       2	@?P?%???@?P?%???!@?P?%???:	??*??O@??*??O@!??*??O@B      ??!       J	?,^,???,^,??!?,^,??R      ??!       Z	?,^,???,^,??!?,^,??b      ??!       JGPUY?????b q??&'?PX@y?7??@?"<
sequential_21/dense_416/MatMulMatMul????????!????????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?s)i??!_M2????"J
,gradient_tape/sequential_21/dense_416/MatMulMatMulO??*????!&?=?5??0"-
IteratorGetNext/_3_Send?m2?????!?m?<???"1
Nadam/Nadam/update/addAddV2??????v?!ۈ???|??"3
Nadam/Nadam/update/add_2AddV2??????v?!Rh8i???"9
Nadam/Nadam/update/truediv_3RealDive???r?!j?4?tֱ?"3
Nadam/Nadam/update/add_1AddV2"?e??xq?!?%?V???"<
sequential_21/dense_436/SoftmaxSoftmax"?e??xq?!.??Ñ??"/
Nadam/Nadam/update/subSub?x?_Xap?!?z?I???Q      Y@Y#%?T?/??ako??@?X@q??'???W@y?V?w?y??"?
both?Your program is POTENTIALLY input-bound because 94.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 