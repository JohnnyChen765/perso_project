?	?P???^K@?P???^K@!?P???^K@	?6???????6??????!?6??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?P???^K@???˯I@1?? ????A͐*?WY??I??v?@Y????(y??*	gffffvS@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg)YNB??!?;?Oo??@)?ڧ?1??1?g???[:@:Preprocessing2U
Iterator::Model::ParallelMapV2?v???!?S???6@)?v???1?S???6@:Preprocessing2F
Iterator::Model??mnLO??!??o5?uD@)`??-??1S???L2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????~?!lU?h#@)?????~?1lU?h#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM?^?iN??!k[?/#3@)???֪}?1h7?F??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!\??p?M@)?Ɵ?lXs?1Wm??eD@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr1?q?p?!?Oo??N@)r1?q?p?1?Oo??N@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??̯? ??!L???XT5@)?	?y?]?1/?ͬ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?6??????IA$?g?X@QPl?K'??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???˯I@???˯I@!???˯I@      ??!       "	?? ?????? ????!?? ????*      ??!       2	͐*?WY??͐*?WY??!͐*?WY??:	??v?@??v?@!??v?@B      ??!       J	????(y??????(y??!????(y??R      ??!       Z	????(y??????(y??!????(y??b      ??!       JGPUY?6??????b qA$?g?X@yPl?K'???"I
+gradient_tape/sequential_11/dense_34/MatMulMatMulG?c3??!G?c3??0"I
+gradient_tape/sequential_11/dense_33/MatMulMatMul?????!\??>???0";
sequential_11/dense_33/MatMulMatMulu?ha???!?(?7{ɠ?0";
sequential_11/dense_34/MatMulMatMulu?ha???!hS'P~??0"I
-gradient_tape/sequential_11/dense_34/MatMul_1MatMul?????!lX?_G??"I
-gradient_tape/sequential_11/dense_35/MatMul_1MatMulH???.a}?!?N??????"Y
8gradient_tape/sequential_11/dense_35/BiasAdd/BiasAddGradBiasAddGradz???.y?!??P????"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchEKc?-y?!?##????";
sequential_11/dense_35/MatMulMatMulEKc?-y?!TX?dk2??0"e
>sequential_11/batch_normalization_22/moments/SquaredDifferenceSquaredDifference???t?!??W?#???Q      Y@Y?? _??@af?*??W@q[SK?U@yΒ-?ͫ??"?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?87.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 