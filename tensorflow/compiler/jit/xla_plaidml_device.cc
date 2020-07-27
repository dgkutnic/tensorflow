/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Registers the XLA_PLAIDML device, which is an XlaDevice instantiation that runs
// operators using XLA via the XLA "Host" (PLAIDML) backend.

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"
// #include "tensorflow/core/framework/kernel_def.pb.h"
// #include "tensorflow/compiler/tf2xla/tf2xla_util.h"

namespace tensorflow {

class XlaPlaidMLDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaPlaidMLDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices) {
    LOG(INFO) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return Status::OK();
  }

  devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_PLAIDML, ":0"));
  return Status::OK();
}

Status XlaPlaidMLDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices) {
    LOG(INFO) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return Status::OK();
  }
  bool compile_on_demand = flags->tf_xla_compile_on_demand;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_PLAIDML_XLA_JIT;
  registration.autoclustering_policy =
      compile_on_demand
          ? XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested
          : XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_PLAIDML, registration);

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_PLAIDML, DEVICE_PLAIDML_XLA_JIT);
  (void)registrations;

  TF_ASSIGN_OR_RETURN(auto platform,
                      se::MultiPlatformManager::PlatformWithName("Host"));

  XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_XLA_PLAIDML;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_PLAIDML_XLA_JIT;
  options.use_multiple_streams = false;
  auto device = absl::make_unique<XlaDevice>(session_options, options);

  devices->push_back(std::move(device));
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_PLAIDML, XlaPlaidMLDeviceFactory, 250);

// Kernel registrations

constexpr std::array<DataType, 16> kAllXlaPlaidMLTypes = {
    {DT_UINT8, DT_QUINT8, DT_UINT16, DT_INT8, DT_QINT8, DT_INT16, DT_INT32,
     DT_QINT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
     DT_COMPLEX128, DT_BOOL, DT_BFLOAT16}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_PLAIDML, XlaLocalLaunchOp, kAllXlaPlaidMLTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_PLAIDML, XlaCompileOp, kAllXlaPlaidMLTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_PLAIDML, XlaRunOp, kAllXlaPlaidMLTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_PLAIDML, kAllXlaPlaidMLTypes);

}  // namespace tensorflow
