// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<uint8_t> ChannelGenerator::findFirstAvailableChannel(
    uint8_t numChannels,
    ArrayRef<llvm::SmallSetVector<uint8_t, 8>> excludeSets,
    ArrayRef<uint8_t> order) {
  auto isAvailable = [&](uint8_t channel) {
    return llvm::none_of(
        excludeSets, [&](const llvm::SmallSetVector<uint8_t, 8> &excludeSet) {
          return excludeSet.count(channel);
        });
  };
  if (!order.empty()) {
    for (uint8_t channel : order) {
      if (channel < numChannels && isAvailable(channel)) return channel;
    }
    return std::nullopt;
  }
  for (uint8_t channel = 0; channel < numChannels; ++channel) {
    if (isAvailable(channel)) return channel;
  }
  return std::nullopt;
}

std::optional<uint8_t> ChannelGenerator::getAndAssignProducerDMAChannel(
    ChannelAssignmentMode mode, ArrayRef<uint8_t> preferredOrder) {
  std::optional<uint8_t> channel;
  switch (mode) {
    case ChannelAssignmentMode::FirstAvailableCircuitFlow: {
      // Select the first available channel for circuit flow.
      // A channel is valid if it is not already assigned to any circuit or
      // packet flow.
      channel = findFirstAvailableChannel(
          numProducerChannels,
          {assignedCircuitProducerChannels, assignedPacketProducerChannels});
      break;
    }
    case ChannelAssignmentMode::FirstAvailablePacketFlow: {
      // Select the first available channel for packet flow.
      // A channel is valid if it is not already assigned to a circuit flow.
      channel =
          findFirstAvailableChannel(numProducerChannels,
                                    {assignedCircuitProducerChannels},
                                    preferredOrder);
      break;
    }
    case ChannelAssignmentMode::RoundRobinPacketFlow: {
      // Select the channel for packet flow, using a round-robin strategy for
      // load balancing:
      // 1. Prefer an unused channel (not assigned to any circuit or packet
      // flow), trying `preferredOrder` first if given.
      // 2. If no such channel is available, reuse a packet flow channel:
      // the first one in `preferredOrder` that's currently in use, or (with
      // no preference given, or none of it in use) the least recently used
      // one from `assignedPacketProducerChannels.front()`.
      channel = findFirstAvailableChannel(
          numProducerChannels,
          {assignedCircuitProducerChannels, assignedPacketProducerChannels},
          preferredOrder);
      if (!channel) {
        for (uint8_t candidate : preferredOrder) {
          if (assignedPacketProducerChannels.count(candidate)) {
            channel = candidate;
            break;
          }
        }
      }
      if (!channel && !assignedPacketProducerChannels.empty())
        channel = assignedPacketProducerChannels.front();
      break;
    }
    default:
      assert(false && "Unsupported ChannelAssignmentMode");
  }
  // Assign the channel if found.
  if (channel.has_value()) assignProducerDMAChannel(channel.value(), mode);
  return channel;
}

std::optional<uint8_t> ChannelGenerator::getAndAssignConsumerDMAChannel(
    ChannelAssignmentMode mode, ArrayRef<uint8_t> preferredOrder) {
  std::optional<uint8_t> channel;
  switch (mode) {
    case ChannelAssignmentMode::FirstAvailableCircuitFlow: {
      // Select the first available channel for circuit flow.
      // A channel is valid if it is not already assigned to any circuit or
      // packet flow.
      channel = findFirstAvailableChannel(
          numConsumerChannels,
          {assignedCircuitConsumerChannels, assignedPacketConsumerChannels});
      break;
    }
    case ChannelAssignmentMode::FirstAvailablePacketFlow: {
      // Select the first available channel for packet flow.
      // A channel is valid if it is not already assigned to a circuit flow.
      channel =
          findFirstAvailableChannel(numConsumerChannels,
                                    {assignedCircuitConsumerChannels},
                                    preferredOrder);
      break;
    }
    case ChannelAssignmentMode::RoundRobinPacketFlow: {
      // Select the channel for packet flow, using a round-robin strategy for
      // load balancing:
      // 1. Prefer an unused channel (not assigned to any circuit or packet
      // flow), trying `preferredOrder` first if given.
      // 2. If no such channel is available, reuse a packet flow channel:
      // the first one in `preferredOrder` that's currently in use, or (with
      // no preference given, or none of it in use) the least recently used
      // one from `assignedPacketConsumerChannels.front()`.
      channel = findFirstAvailableChannel(
          numConsumerChannels,
          {assignedCircuitConsumerChannels, assignedPacketConsumerChannels},
          preferredOrder);
      if (!channel) {
        for (uint8_t candidate : preferredOrder) {
          if (assignedPacketConsumerChannels.count(candidate)) {
            channel = candidate;
            break;
          }
        }
      }
      if (!channel && !assignedPacketConsumerChannels.empty())
        channel = assignedPacketConsumerChannels.front();
      break;
    }
    default:
      assert(false && "Unsupported ChannelAssignmentMode");
  }
  // Assign the channel if found.
  if (channel.has_value()) assignConsumerDMAChannel(channel.value(), mode);
  return channel;
}

void ChannelGenerator::assignProducerDMAChannel(uint8_t channel,
                                                ChannelAssignmentMode mode) {
  switch (mode) {
    case ChannelAssignmentMode::FirstAvailableCircuitFlow:
      assignedCircuitProducerChannels.insert(channel);
      break;
    case ChannelAssignmentMode::FirstAvailablePacketFlow:
      assignedPacketProducerChannels.insert(channel);
      break;
    case ChannelAssignmentMode::RoundRobinPacketFlow:
      // Remove and reinsert to update the least recently used channel
      // (front).
      assignedPacketProducerChannels.remove(channel);
      assignedPacketProducerChannels.insert(channel);
      break;
    default:
      assert(false && "Unsupported ChannelAssignmentMode");
  }
}

void ChannelGenerator::assignConsumerDMAChannel(uint8_t channel,
                                                ChannelAssignmentMode mode) {
  switch (mode) {
    case ChannelAssignmentMode::FirstAvailableCircuitFlow:
      assignedCircuitConsumerChannels.insert(channel);
      break;
    case ChannelAssignmentMode::FirstAvailablePacketFlow:
      assignedPacketConsumerChannels.insert(channel);
      break;
    case ChannelAssignmentMode::RoundRobinPacketFlow:
      // Remove and reinsert to update the least recently used channel
      // (front).
      assignedPacketConsumerChannels.remove(channel);
      assignedPacketConsumerChannels.insert(channel);
      break;
    default:
      assert(false && "Unsupported ChannelAssignmentMode");
  }
}

}  // namespace mlir::iree_compiler::AMDAIE
