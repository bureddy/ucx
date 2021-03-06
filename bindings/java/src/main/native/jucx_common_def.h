/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef HELPER_H_
#define HELPER_H_

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>

#include <jni.h>

#include <cstdint>


typedef uintptr_t native_ptr;

static void log_error(const char* error);

JNIEXPORT void JNICALL JNU_ThrowException(JNIEnv *, const char *);

void JNU_ThrowExceptionByStatus(JNIEnv *, ucs_status_t);

#define JUCX_DEFINE_LONG_CONSTANT(_name) do { \
    jfieldID field = env->GetStaticFieldID(cls, #_name, "J"); \
    env->SetStaticLongField(cls, field, _name); \
} while(0)

/**
 * @brief Utility to convert Java InetSocketAddress class (corresponds to the Network Layer 4
 * and consists of an IP address and a port number) to corresponding sockaddr_storage struct.
 * Supports IPv4 and IPv6.
 */
bool j2cInetSockAddr(JNIEnv *env, jobject sock_addr, sockaddr_storage& ss, socklen_t& sa_len);

#endif
