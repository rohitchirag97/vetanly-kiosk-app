#pragma once

#include "EnrollmentMatcher.hpp"

#include <NitroModules/ArrayBuffer.hpp>

#include <memory>

namespace margelo::nitro::facerecognizer {

/** Serializes normalized embedding samples into a JS-owned ArrayBuffer. */
std::shared_ptr<margelo::nitro::ArrayBuffer> serializeEnrollment(
    const EnrollmentSamples& samples);

/**
 * Deserializes a JS ArrayBuffer into normalized embedding samples.
 *
 * Throws when the payload is malformed so persisted data corruption is visible
 * to the caller instead of silently weakening recognition.
 */
EnrollmentSamples deserializeEnrollment(
    const std::shared_ptr<margelo::nitro::ArrayBuffer>& serializedEnrollment);

}  // namespace margelo::nitro::facerecognizer
