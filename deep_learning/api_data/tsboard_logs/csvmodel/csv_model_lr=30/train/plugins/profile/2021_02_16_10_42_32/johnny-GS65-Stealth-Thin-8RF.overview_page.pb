?	???%?O@???%?O@!???%?O@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???%?O@й??҄M@1????i???A??J\Ǹ??I7????@*	#??~jtS@2U
Iterator::Model::ParallelMapV2q ?????!?t?X8?7@)q ?????1?t?X8?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!???;@)??q?????1????>6@:Preprocessing2F
Iterator::Modeln?2d???!?9?zF@)jhwH1??1?????Q4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????v??!?c#Ύ?5@)ȔA????1;?
0?~+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>Z?1?	z?!??;l?V @)>Z?1?	z?1??;l?V @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?A?<?E??!7?y???K@)<??~Kq?1 ?f???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~?o?!?????x@)?~?o?1?????x@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8??@???!??mM?[7@)E?4fR?1<?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?"?I ?X@QvN??????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	й??҄M@й??҄M@!й??҄M@      ??!       "	????i???????i???!????i???*      ??!       2	??J\Ǹ????J\Ǹ??!??J\Ǹ??:	7????@7????@!7????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?"?I ?X@yvN???????";
sequential_26/dense_78/MatMulMatMulh??0???!h??0???0"I
+gradient_tape/sequential_26/dense_78/MatMulMatMul???????!??Bd???0"I
+gradient_tape/sequential_26/dense_79/MatMulMatMul???????!???މ??0";
sequential_26/dense_79/MatMulMatMul???????!@?y:??0"I
-gradient_tape/sequential_26/dense_79/MatMul_1MatMul˅sc???!?Ȗң???"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchR?m?z?!??Xt1I??"I
-gradient_tape/sequential_26/dense_80/MatMul_1MatMulH?[z?z?!???_PM??";
sequential_26/dense_80/MatMulMatMulH?[z?z?!o????0"Y
8gradient_tape/sequential_26/dense_80/BiasAdd/BiasAddGradBiasAddGrad?#cH?v?!C?1??c??"e
>sequential_26/batch_normalization_52/moments/SquaredDifferenceSquaredDifference?????v?!k???е?Q      Y@Yg<??x@a9?as?W@q??5?ZX@y??"=???"?
both?Your program is POTENTIALLY input-bound because 93.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 