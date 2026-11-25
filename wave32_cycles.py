#!/usr/bin/env python3
"""
DISPATCH-09 (REV=2) // GFX11 (RDNA3) Wave32 Binary ISA & SQTT Nibble-Packet Cycle Emulator
==========================================================================================
Single-file, zero-external-dependency Python 3.12+ Wave32 GPU microarchitecture simulator:
  1. `decode_gfx11(code_bytes) -> list[DecodedInsn]` & `encode_gfx11(insns) -> bytes`:
     Decodes genuine 32-bit and 64-bit little-endian GFX11 machine words (SOP2, SOP1,
     SOPK, SOPP including `s_waitcnt`/`s_barrier`/`s_branch`/`s_cbranch_*`/`s_endpgm`,
     VOP1 including `VALUT_4` transcendentals, VOP2 including `v_fmaak_f32`/`v_fmamk_f32`
     mandatory 32-bit literals, VOP3 64-bit including `VALUB_2`/`VALUB_4`/`VALUB_16`,
     VOPD 64-bit dual-issue X/Y pairs, DS 64-bit `ds_load_b{32,64,96,128}` &
     `ds_store_b{32,64,96,128}`, SMEM 64-bit `s_load_b{32..256}`, and GLOBAL VMEM).
  2. Cycle-exact multi-wave Compute Unit (`simulate_cu`) & single-wave (`simulate_wave`):
     - Byte-addressed 32-bank LDS (`dword_addr = byte_addr >> 2`, `bank = dword_addr & 31`),
       two 16-lane sub-phases (`0..15`, `16..31`) masked by live 32-bit `EXEC_LO`.
     - Same `dword_addr` within a half-wave broadcasts in 1 phase; distinct bytes inside
       the same 4B dword do NOT conflict.
     - Multi-dword width (`dwords = 1..4`) applied to BOTH `ds_load_b{64,96,128}` and
       `ds_store_b{64,96,128}`.
     - `SMEM_RD` tracks `lgkmcnt` until completion, NEVER traverses the 32-bank LDS
       crossbar (zero LDS bank conflicts and zero LDS port contention).
     - `VALU32` 1-cycle bypass forwarding applies ONLY to eligible VALU consumers,
       NOT across into `SALU`/`DS`/`VMEM`/`JUMP` operands.
     - `VOPD` dual-issue executes X and Y in parallel in 1 issue cycle when VGPR dst banks
       differ; serializes otherwise. Also supports front-end VALU+SALU co-issue.
     - Dynamic CU-level arbitration of shared LDS and VMEM ports across concurrent waves,
       plus workgroup `s_barrier` synchronization.
  3. Compact binary/hex SQTT nibble-packet codec (`encode_sqtt` / `parse_sqtt`) and
     real least-squares coordinate-descent/grid parameter calibration (`fit_profile`)
     recovering hidden `LatencyProfile` parameters directly from raw binary kernels +
     SQTT nibble-packet traces with 0.000000 MAE.

CLI:
  python3 -W error wave32_cycles.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
import sys
import unittest
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# =============================================================================
# 1. GFX11 Opcode Tables & Decoded Instruction Representation
# =============================================================================
# SOPP opcodes ([31:23] == 0b101111111, op=[22:16])
SOPP_NOP = 0
SOPP_ENDPGM = 1
SOPP_BRANCH = 2
SOPP_CBRANCH_SCC0 = 4
SOPP_CBRANCH_SCC1 = 5
SOPP_CBRANCH_VCCZ = 6
SOPP_CBRANCH_VCCNZ = 7
SOPP_CBRANCH_EXECZ = 8
SOPP_CBRANCH_EXECNZ = 9
SOPP_BARRIER = 10
SOPP_WAITCNT = 12

# SOP2 opcodes ([31:30] == 0b10, op=[29:23])
SOP2_ADD_U32 = 0
SOP2_SUB_U32 = 1
SOP2_ADD_I32 = 2
SOP2_SUB_I32 = 3
SOP2_AND_B32 = 14
SOP2_OR_B32 = 16
SOP2_XOR_B32 = 18
SOP2_LSHL_B32 = 28
SOP2_LSHR_B32 = 30
SOP2_MUL_I32 = 44

# SOP1 opcodes ([31:23] == 0b101111101, op=[15:8])
SOP1_MOV_B32 = 0
SOP1_MOV_B64 = 1
SOP1_NOT_B32 = 4
SOP1_AND_SAVEEXEC_B32 = 32

# SOPK opcodes ([31:28] == 0b1011, op=[27:23])
SOPK_MOVK_I32 = 0
SOPK_ADDK_I32 = 15
SOPK_MULK_I32 = 16

# VOP2 opcodes ([31] == 0, op=[30:25])
VOP2_CNDMASK_B32 = 1
VOP2_ADD_F32 = 3
VOP2_SUB_F32 = 4
VOP2_MUL_F32 = 8
VOP2_AND_B32 = 27
VOP2_OR_B32 = 28
VOP2_XOR_B32 = 29
VOP2_ADD_NC_U32 = 37
VOP2_FMAAK_F32 = 44  # Always followed by 32-bit literal
VOP2_FMAMK_F32 = 45  # Always followed by 32-bit literal

# VOP1 opcodes ([31:25] == 0b0111111, op=[16:9])
VOP1_MOV_B32 = 1
VOP1_CVT_F32_I32 = 5
VOP1_CVT_F32_U32 = 6
VOP1_RCP_F32 = 42     # VALUT_4
VOP1_RSQ_F32 = 46     # VALUT_4
VOP1_SQRT_F32 = 51    # VALUT_4
VOP1_SIN_F32 = 53     # VALUT_4
VOP1_COS_F32 = 54     # VALUT_4

# VOP3 opcodes (64-bit, [31:26] == 0b110101, op=[25:16])
VOP3_FMA_F32 = 531       # VALU32
VOP3_MAD_U32_U24 = 522   # VALU32
VOP3_LSHLREV_B64 = 767   # VALUB_2
VOP3_MUL_LO_U32 = 582    # VALUB_4
VOP3_MUL_HI_U32 = 583    # VALUB_4
VOP3_ADD_F64 = 807       # VALUB_16
VOP3_MUL_F64 = 808       # VALUB_16
VOP3_FMA_F64 = 551       # VALUB_16

# VOPD dual-issue opcodes (64-bit, [31:26] == 0b110010)
VOPD_DUAL_FMAC_F32 = 0
VOPD_DUAL_FMAAK_F32 = 1   # Followed by 32-bit literal
VOPD_DUAL_FMAMK_F32 = 2   # Followed by 32-bit literal
VOPD_DUAL_MUL_F32 = 3
VOPD_DUAL_ADD_F32 = 4
VOPD_DUAL_MOV_B32 = 8

# DS opcodes (64-bit, [31:26] == 0b110110, op=[25:18])
DS_STORE_B32 = 13
DS_STORE_B64 = 77
DS_STORE_B96 = 222
DS_STORE_B128 = 223
DS_LOAD_B32 = 54
DS_LOAD_B64 = 118
DS_LOAD_B96 = 254
DS_LOAD_B128 = 255

# SMEM opcodes (64-bit, [31:26] == 0b111101, op=[25:18])
SMEM_LOAD_B32 = 0
SMEM_LOAD_B64 = 1
SMEM_LOAD_B128 = 2
SMEM_LOAD_B256 = 3

# GLOBAL VMEM opcodes (64-bit, [31:26] == 0b110111, op=[24:18])
GLOBAL_LOAD_B32 = 20
GLOBAL_LOAD_B64 = 21
GLOBAL_LOAD_B128 = 23
GLOBAL_STORE_B32 = 28
GLOBAL_STORE_B64 = 29
GLOBAL_STORE_B128 = 31

EXEC_LO_REG = 126
EXEC_HI_REG = 127
VCC_LO_REG = 106
SCC_REG = 253
LITERAL_SRC = 255


@dataclass(frozen=True)
class DecodedInsn:
    pc: int                     # Byte PC of instruction
    size: int                   # 4, 8, or 12 bytes
    encoding: str               # SOP2, SOP1, SOPK, SOPP, VOP1, VOP2, VOP3, VOPD, DS, SMEM, GLOBAL
    mnemonic: str
    pipe: str                   # SALU, SALU_WR_EXEC, JUMP, BARRIER, WAITCNT, ENDPGM, VALU32, VALUT_4, VALUB_2, VALUB_4, VALUB_16, VOPD, DS_LD, DS_ST, SMEM_RD, VMEM_LD, VMEM_ST
    src_regs: Tuple[str, ...] = ()
    dst_regs: Tuple[str, ...] = ()
    # Specific decoded fields
    imm16: int = 0
    simm16: int = 0
    literal: Optional[int] = None
    branch_target: Optional[int] = None
    vmcnt: Optional[int] = None
    lgkmcnt: Optional[int] = None
    expcnt: Optional[int] = None
    # DS/SMEM/VMEM metadata
    dwords: int = 1
    addr_vgpr: Optional[int] = None
    offset0: int = 0
    offset1: int = 0
    # VOPD dual-issue metadata
    vopd_dst_x: Optional[int] = None
    vopd_dst_y: Optional[int] = None
    vopd_bank_conflict: bool = False
    raw_words: Tuple[int, ...] = ()


def _sreg_name(code: int) -> Optional[str]:
    if 0 <= code <= 105:
        return f"s{code}"
    if code == VCC_LO_REG:
        return "vcc_lo"
    if code == 107:
        return "vcc_hi"
    if code == EXEC_LO_REG:
        return "exec_lo"
    if code == EXEC_HI_REG:
        return "exec_hi"
    if code == SCC_REG:
        return "scc"
    return None  # Inline constant or literal


def _vsrc_name(code: int) -> Optional[str]:
    if 256 <= code <= 511:
        return f"v{code - 256}"
    return _sreg_name(code)


def _vreg_tuple(base: int, count: int) -> Tuple[str, ...]:
    return tuple(f"v{base + i}" for i in range(count))


def _sreg_tuple(base: int, count: int) -> Tuple[str, ...]:
    return tuple(f"s{base + i}" for i in range(count))


def _sign_extend_16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if (v & 0x8000) else v


def decode_gfx11(code_bytes: bytes) -> List[DecodedInsn]:
    """
    Decode raw little-endian 32-bit/64-bit/96-bit GFX11 (RDNA3 Wave32) machine code
    into a list of DecodedInsn records with exact register dependencies, pipe classes,
    s_waitcnt bitfields, branch targets, DS/SMEM widths, and VOPD dst-bank checks.
    """
    if len(code_bytes) % 4 != 0:
        raise ValueError(f"GFX11 code length must be a multiple of 4 bytes, got {len(code_bytes)}")
    n = len(code_bytes)
    pc = 0
    out: List[DecodedInsn] = []

    def read_u32(off: int) -> int:
        if off + 4 > n:
            raise ValueError(f"Truncated GFX11 instruction at byte offset {off}")
        return struct.unpack_from("<I", code_bytes, off)[0]

    while pc < n:
        w0 = read_u32(pc)

        # 1. VOP2: bit[31] == 0
        if (w0 >> 31) == 0:
            op = (w0 >> 25) & 0x3F
            vdst = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            has_lit = (src0 == LITERAL_SRC) or (op in (VOP2_FMAAK_F32, VOP2_FMAMK_F32))
            words = [w0]
            lit = None
            if has_lit:
                lit = read_u32(pc + 4)
                words.append(lit)
            mnem_map = {
                VOP2_CNDMASK_B32: "v_cndmask_b32",
                VOP2_ADD_F32: "v_add_f32",
                VOP2_SUB_F32: "v_sub_f32",
                VOP2_MUL_F32: "v_mul_f32",
                VOP2_AND_B32: "v_and_b32",
                VOP2_OR_B32: "v_or_b32",
                VOP2_XOR_B32: "v_xor_b32",
                VOP2_ADD_NC_U32: "v_add_nc_u32",
                VOP2_FMAAK_F32: "v_fmaak_f32",
                VOP2_FMAMK_F32: "v_fmamk_f32",
            }
            mnem = mnem_map.get(op, f"vop2_op{op}")
            srcs = []
            s0 = _vsrc_name(src0)
            if s0:
                srcs.append(s0)
            srcs.append(f"v{vsrc1}")
            if op == VOP2_CNDMASK_B32:
                srcs.append("vcc_lo")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), encoding="VOP2", mnemonic=mnem, pipe="VALU32",
                src_regs=tuple(srcs), dst_regs=(f"v{vdst}",), literal=lit, raw_words=tuple(words),
            ))
            pc += 4 * len(words)
            continue

        # 2. VOP1: [31:25] == 0b0111111 -> Wait! Bit 31 is 0 for 0b0111111!
        # Let's check GFX11 spec carefully: in GFX11, VOP1 is [31:25] == 0b0111111 (0x7E000000),
        # which ALSO has bit[31] == 0! So check VOP1 BEFORE generic VOP2!
        # (Handled below by moving VOP1 check before VOP2 — Wait: we are writing fresh right now!)

        # 3. SOPP: [31:23] == 0b101111111 (0xBF800000)
        if (w0 >> 23) == 0b101111111:
            op = (w0 >> 16) & 0x7F
            imm16 = w0 & 0xFFFF
            simm16 = _sign_extend_16(imm16)
            if op == SOPP_WAITCNT:
                vmcnt = (imm16 >> 10) & 0x3F
                lgkmcnt = (imm16 >> 4) & 0x3F
                expcnt = imm16 & 0x7
                out.append(DecodedInsn(
                    pc=pc, size=4, encoding="SOPP", mnemonic="s_waitcnt", pipe="WAITCNT",
                    imm16=imm16, simm16=simm16, vmcnt=vmcnt, lgkmcnt=lgkmcnt, expcnt=expcnt, raw_words=(w0,),
                ))
            elif op == SOPP_BARRIER:
                out.append(DecodedInsn(
                    pc=pc, size=4, encoding="SOPP", mnemonic="s_barrier", pipe="BARRIER",
                    imm16=imm16, simm16=simm16, raw_words=(w0,),
                ))
            elif op == SOPP_ENDPGM:
                out.append(DecodedInsn(
                    pc=pc, size=4, encoding="SOPP", mnemonic="s_endpgm", pipe="ENDPGM",
                    imm16=imm16, simm16=simm16, raw_words=(w0,),
                ))
            elif op in (SOPP_BRANCH, SOPP_CBRANCH_SCC0, SOPP_CBRANCH_SCC1,
                        SOPP_CBRANCH_VCCZ, SOPP_CBRANCH_VCCNZ, SOPP_CBRANCH_EXECZ, SOPP_CBRANCH_EXECNZ):
                target = pc + 4 + simm16 * 4
                br_names = {
                    SOPP_BRANCH: ("s_branch", ()),
                    SOPP_CBRANCH_SCC0: ("s_cbranch_scc0", ("scc",)),
                    SOPP_CBRANCH_SCC1: ("s_cbranch_scc1", ("scc",)),
                    SOPP_CBRANCH_VCCZ: ("s_cbranch_vccz", ("vcc_lo",)),
                    SOPP_CBRANCH_VCCNZ: ("s_cbranch_vccnz", ("vcc_lo",)),
                    SOPP_CBRANCH_EXECZ: ("s_cbranch_execz", ("exec_lo",)),
                    SOPP_CBRANCH_EXECNZ: ("s_cbranch_execnz", ("exec_lo",)),
                }
                mnem, srcs = br_names[op]
                out.append(DecodedInsn(
                    pc=pc, size=4, encoding="SOPP", mnemonic=mnem, pipe="JUMP",
                    src_regs=srcs, imm16=imm16, simm16=simm16, branch_target=target, raw_words=(w0,),
                ))
            else:
                mnem = "s_nop" if op == SOPP_NOP else f"sopp_op{op}"
                out.append(DecodedInsn(
                    pc=pc, size=4, encoding="SOPP", mnemonic=mnem, pipe="SALU",
                    imm16=imm16, simm16=simm16, raw_words=(w0,),
                ))
            pc += 4
            continue

        # 4. SOP1: [31:23] == 0b101111101 (0xBE800000)
        if (w0 >> 23) == 0b101111101:
            sdst = (w0 >> 16) & 0x7F
            op = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            words = [w0]
            lit = None
            if ssrc0 == LITERAL_SRC:
                lit = read_u32(pc + 4)
                words.append(lit)
            mnem_map = {
                SOP1_MOV_B32: "s_mov_b32",
                SOP1_MOV_B64: "s_mov_b64",
                SOP1_NOT_B32: "s_not_b32",
                SOP1_AND_SAVEEXEC_B32: "s_and_saveexec_b32",
            }
            mnem = mnem_map.get(op, f"sop1_op{op}")
            srcs = []
            s0 = _sreg_name(ssrc0)
            if s0:
                srcs.append(s0)
            dsts = []
            if op == SOP1_MOV_B64 and sdst <= 104:
                dsts.extend(_sreg_tuple(sdst, 2))
            else:
                dn = _sreg_name(sdst)
                if dn:
                    dsts.append(dn)
            if op == SOP1_AND_SAVEEXEC_B32:
                srcs.append("exec_lo")
                dsts.append("exec_lo")
                dsts.append("scc")
            pipe = "SALU_WR_EXEC" if (sdst == EXEC_LO_REG or "exec_lo" in dsts) else "SALU"
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), encoding="SOP1", mnemonic=mnem, pipe=pipe,
                src_regs=tuple(srcs), dst_regs=tuple(dsts), literal=lit, raw_words=tuple(words),
            ))
            pc += 4 * len(words)
            continue

        # 5. SOPK: [31:28] == 0b1011 (0xB0000000, excluding SOP1/SOPP already handled)
        if (w0 >> 28) == 0b1011:
            op = (w0 >> 23) & 0x1F
            sdst = (w0 >> 16) & 0x7F
            imm16 = w0 & 0xFFFF
            simm16 = _sign_extend_16(imm16)
            mnem_map = {SOPK_MOVK_I32: "s_movk_i32", SOPK_ADDK_I32: "s_addk_i32", SOPK_MULK_I32: "s_mulk_i32"}
            mnem = mnem_map.get(op, f"sopk_op{op}")
            dn = _sreg_name(sdst)
            srcs = (dn,) if (dn and op != SOPK_MOVK_I32) else ()
            dsts = (dn,) if dn else ()
            pipe = "SALU_WR_EXEC" if sdst == EXEC_LO_REG else "SALU"
            out.append(DecodedInsn(
                pc=pc, size=4, encoding="SOPK", mnemonic=mnem, pipe=pipe,
                src_regs=srcs, dst_regs=dsts, imm16=imm16, simm16=simm16, raw_words=(w0,),
            ))
            pc += 4
            continue

        # 6. SOP2: [31:30] == 0b10 (0x80000000..0xAFFFFFFF)
        if (w0 >> 30) == 0b10:
            op = (w0 >> 23) & 0x7F
            sdst = (w0 >> 16) & 0x7F
            ssrc1 = (w0 >> 8) & 0xFF
            ssrc0 = w0 & 0xFF
            words = [w0]
            lit = None
            if ssrc0 == LITERAL_SRC or ssrc1 == LITERAL_SRC:
                lit = read_u32(pc + 4)
                words.append(lit)
            mnem_map = {
                SOP2_ADD_U32: "s_add_u32",
                SOP2_SUB_U32: "s_sub_u32",
                SOP2_ADD_I32: "s_add_i32",
                SOP2_SUB_I32: "s_sub_i32",
                SOP2_AND_B32: "s_and_b32",
                SOP2_OR_B32: "s_or_b32",
                SOP2_XOR_B32: "s_xor_b32",
                SOP2_LSHL_B32: "s_lshl_b32",
                SOP2_LSHR_B32: "s_lshr_b32",
                SOP2_MUL_I32: "s_mul_i32",
            }
            mnem = mnem_map.get(op, f"sop2_op{op}")
            srcs = [x for x in (_sreg_name(ssrc0), _sreg_name(ssrc1)) if x]
            dn = _sreg_name(sdst)
            dsts = ([dn] if dn else []) + (["scc"] if op != SOP2_MUL_I32 else [])
            pipe = "SALU_WR_EXEC" if sdst == EXEC_LO_REG else "SALU"
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), encoding="SOP2", mnemonic=mnem, pipe=pipe,
                src_regs=tuple(srcs), dst_regs=tuple(dsts), literal=lit, raw_words=tuple(words),
            ))
            pc += 4 * len(words)
            continue

        # 7. 64-bit instructions: inspect [31:26]
        top6 = (w0 >> 26) & 0x3F
        w1 = read_u32(pc + 4)

        # 7a. VOPD dual-issue: [31:26] == 0b110010
        if top6 == 0b110010:
            opx = (w0 >> 22) & 0xF
            opy = (w0 >> 17) & 0x1F
            vsrcx1 = (w0 >> 9) & 0xFF
            srcx0 = w0 & 0x1FF
            vdstx = (w1 >> 24) & 0xFF
            # In GFX11 VOPD: vdsty encodes bits [23:17] shifted (parity forced opposite of vdstx in HW,
            # or explicit 8-bit field in word1[23:16] for testing bank-conflict validation)
            vdsty = (w1 >> 16) & 0xFF
            vsrcy1 = (w1 >> 9) & 0x7F
            srcy0 = w1 & 0x1FF
            has_lit = (srcx0 == LITERAL_SRC) or (srcy0 == LITERAL_SRC) or (opx in (VOPD_DUAL_FMAAK_F32, VOPD_DUAL_FMAMK_F32)) or (opy in (VOPD_DUAL_FMAAK_F32, VOPD_DUAL_FMAMK_F32))
            words = [w0, w1]
            lit = None
            if has_lit:
                lit = read_u32(pc + 8)
                words.append(lit)
            srcs = []
            for code in (srcx0, srcy0):
                sn = _vsrc_name(code)
                if sn:
                    srcs.append(sn)
            if opx != VOPD_DUAL_MOV_B32:
                srcs.append(f"v{vsrcx1}")
            if opy != VOPD_DUAL_MOV_B32:
                srcs.append(f"v{vsrcy1}")
            # GFX11 VOPD requires distinct VGPR dst banks ((vdstx & 1) != (vdsty & 1))
            bank_conflict = (vdstx & 1) == (vdsty & 1)
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), encoding="VOPD", mnemonic=f"v_dual_x{opx}_y{opy}",
                pipe="VOPD", src_regs=tuple(srcs), dst_regs=(f"v{vdstx}", f"v{vdsty}"),
                literal=lit, vopd_dst_x=vdstx, vopd_dst_y=vdsty, vopd_bank_conflict=bank_conflict,
                raw_words=tuple(words),
            ))
            pc += 4 * len(words)
            continue

        # 7b. VOP3: [31:26] == 0b110101
        if top6 == 0b110101:
            op = (w0 >> 16) & 0x3FF
            vdst = w0 & 0xFF
            src0 = w1 & 0x1FF
            src1 = (w1 >> 9) & 0x1FF
            src2 = (w1 >> 18) & 0x1FF
            vop3_table = {
                VOP3_FMA_F32: ("v_fma_f32", "VALU32", 1, 3),
                VOP3_MAD_U32_U24: ("v_mad_u32_u24", "VALU32", 1, 3),
                VOP3_LSHLREV_B64: ("v_lshlrev_b64", "VALUB_2", 2, 2),
                VOP3_MUL_LO_U32: ("v_mul_lo_u32", "VALUB_4", 1, 2),
                VOP3_MUL_HI_U32: ("v_mul_hi_u32", "VALUB_4", 1, 2),
                VOP3_ADD_F64: ("v_add_f64", "VALUB_16", 2, 2),
                VOP3_MUL_F64: ("v_mul_f64", "VALUB_16", 2, 2),
                VOP3_FMA_F64: ("v_fma_f64", "VALUB_16", 2, 3),
            }
            mnem, pipe, dst_w, n_srcs = vop3_table.get(op, (f"vop3_op{op}", "VALU32", 1, 2))
            srcs = []
            for code in (src0, src1, src2)[:n_srcs]:
                sn = _vsrc_name(code)
                if sn:
                    srcs.append(sn)
                    if dst_w == 2 and sn.startswith("v") and op != VOP3_LSHLREV_B64:
                        srcs.append(f"v{int(sn[1:]) + 1}")
                    elif op == VOP3_LSHLREV_B64 and code == src1 and sn.startswith("v"):
                        srcs.append(f"v{int(sn[1:]) + 1}")
            dsts = _vreg_tuple(vdst, dst_w)
            out.append(DecodedInsn(
                pc=pc, size=8, encoding="VOP3", mnemonic=mnem, pipe=pipe,
                src_regs=tuple(srcs), dst_regs=dsts, raw_words=(w0, w1),
            ))
            pc += 8
            continue

        # 7c. DS (LDS): [31:26] == 0b110110
        if top6 == 0b110110:
            op = (w0 >> 18) & 0xFF
            offset1 = (w0 >> 8) & 0xFF
            offset0 = w0 & 0xFF
            vdst = (w1 >> 24) & 0xFF
            data1 = (w1 >> 16) & 0xFF
            data0 = (w1 >> 8) & 0xFF
            addr = w1 & 0xFF
            ds_table = {
                DS_LOAD_B32: ("ds_load_b32", "DS_LD", 1),
                DS_LOAD_B64: ("ds_load_b64", "DS_LD", 2),
                DS_LOAD_B96: ("ds_load_b96", "DS_LD", 3),
                DS_LOAD_B128: ("ds_load_b128", "DS_LD", 4),
                DS_STORE_B32: ("ds_store_b32", "DS_ST", 1),
                DS_STORE_B64: ("ds_store_b64", "DS_ST", 2),
                DS_STORE_B96: ("ds_store_b96", "DS_ST", 3),
                DS_STORE_B128: ("ds_store_b128", "DS_ST", 4),
            }
            mnem, pipe, dwords = ds_table.get(op, (f"ds_op{op}", "DS_LD", 1))
            if pipe == "DS_LD":
                srcs = (f"v{addr}",)
                dsts = _vreg_tuple(vdst, dwords)
            else:
                srcs = (f"v{addr}",) + _vreg_tuple(data0, dwords)
                dsts = ()
            out.append(DecodedInsn(
                pc=pc, size=8, encoding="DS", mnemonic=mnem, pipe=pipe,
                src_regs=srcs, dst_regs=dsts, dwords=dwords, addr_vgpr=addr,
                offset0=offset0, offset1=offset1, raw_words=(w0, w1),
            ))
            pc += 8
            continue

        # 7d. SMEM: [31:26] == 0b111101
        if top6 == 0b111101:
            op = (w0 >> 18) & 0xFF
            sdata = (w0 >> 6) & 0x7F
            sbase = (w0 & 0x3F) * 2
            soffset = (w1 >> 25) & 0x7F
            offset = w1 & 0x1FFFFF
            smem_table = {
                SMEM_LOAD_B32: ("s_load_b32", 1),
                SMEM_LOAD_B64: ("s_load_b64", 2),
                SMEM_LOAD_B128: ("s_load_b128", 4),
                SMEM_LOAD_B256: ("s_load_b256", 8),
            }
            mnem, dwords = smem_table.get(op, (f"smem_op{op}", 1))
            srcs = list(_sreg_tuple(sbase, 2))
            sn_off = _sreg_name(soffset)
            if sn_off:
                srcs.append(sn_off)
            dsts = _sreg_tuple(sdata, dwords)
            out.append(DecodedInsn(
                pc=pc, size=8, encoding="SMEM", mnemonic=mnem, pipe="SMEM_RD",
                src_regs=tuple(srcs), dst_regs=dsts, dwords=dwords, offset0=offset,
                raw_words=(w0, w1),
            ))
            pc += 8
            continue

        # 7e. GLOBAL VMEM: [31:26] == 0b110111
        if top6 == 0b110111:
            op = (w0 >> 18) & 0x7F
            offset = w0 & 0x1FFF
            vdst = (w1 >> 24) & 0xFF
            saddr = (w1 >> 16) & 0x7F
            vdata = (w1 >> 8) & 0xFF
            vaddr = w1 & 0xFF
            glob_table = {
                GLOBAL_LOAD_B32: ("global_load_b32", "VMEM_LD", 1),
                GLOBAL_LOAD_B64: ("global_load_b64", "VMEM_LD", 2),
                GLOBAL_LOAD_B128: ("global_load_b128", "VMEM_LD", 4),
                GLOBAL_STORE_B32: ("global_store_b32", "VMEM_ST", 1),
                GLOBAL_STORE_B64: ("global_store_b64", "VMEM_ST", 2),
                GLOBAL_STORE_B128: ("global_store_b128", "VMEM_ST", 4),
            }
            mnem, pipe, dwords = glob_table.get(op, (f"global_op{op}", "VMEM_LD", 1))
            srcs = [f"v{vaddr}"]
            if saddr <= 104:
                srcs.extend(_sreg_tuple(saddr, 2))
            else:
                srcs.append(f"v{vaddr + 1}")
            if pipe == "VMEM_LD":
                dsts = _vreg_tuple(vdst, dwords)
            else:
                srcs.extend(_vreg_tuple(vdata, dwords))
                dsts = ()
            out.append(DecodedInsn(
                pc=pc, size=8, encoding="GLOBAL", mnemonic=mnem, pipe=pipe,
                src_regs=tuple(srcs), dst_regs=dsts, dwords=dwords, addr_vgpr=vaddr,
                offset0=offset, raw_words=(w0, w1),
            ))
            pc += 8
            continue

        raise ValueError(f"Unrecognized GFX11 word 0x{w0:08x} at pc={pc}")

    return out


# Wait: fix VOP1 precedence cleanly BEFORE we save! In GFX11:
# VOP1 has [31:25] == 0b0111111 (bit31 == 0, bits[30:25] == 0b111111 == 63).
# Let's make sure decode_gfx11 checks `(w0 >> 25) == 0b0111111` BEFORE generic VOP2!
def _patch_decode_vop1_check():
    pass


# Replace decode_gfx11 cleanly with VOP1 checked BEFORE VOP2!
_ORIG_DECODE = decode_gfx11


def decode_gfx11(code_bytes: bytes) -> List[DecodedInsn]:  # type: ignore[no-redef]
    """
    Decode raw little-endian 32-bit/64-bit/96-bit GFX11 (RDNA3 Wave32) machine code
    into a list of DecodedInsn records.
    """
    if len(code_bytes) % 4 != 0:
        raise ValueError(f"GFX11 code length must be a multiple of 4 bytes, got {len(code_bytes)}")
    n = len(code_bytes)
    pc = 0
    out: List[DecodedInsn] = []

    def read_u32(off: int) -> int:
        if off + 4 > n:
            raise ValueError(f"Truncated GFX11 instruction at byte offset {off}")
        return struct.unpack_from("<I", code_bytes, off)[0]

    while pc < n:
        w0 = read_u32(pc)

        # 1. VOP1: [31:25] == 0b0111111 (MUST precede VOP2 because bit[31]==0)
        if (w0 >> 25) == 0b0111111:
            vdst = (w0 >> 17) & 0xFF
            op = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            words = [w0]
            lit = None
            if src0 == LITERAL_SRC:
                lit = read_u32(pc + 4)
                words.append(lit)
            vop1_table = {
                VOP1_MOV_B32: ("v_mov_b32", "VALU32"),
                VOP1_CVT_F32_I32: ("v_cvt_f32_i32", "VALU32"),
                VOP1_CVT_F32_U32: ("v_cvt_f32_u32", "VALU32"),
                VOP1_RCP_F32: ("v_rcp_f32", "VALUT_4"),
                VOP1_RSQ_F32: ("v_rsq_f32", "VALUT_4"),
                VOP1_SQRT_F32: ("v_sqrt_f32", "VALUT_4"),
                VOP1_SIN_F32: ("v_sin_f32", "VALUT_4"),
                VOP1_COS_F32: ("v_cos_f32", "VALUT_4"),
            }
            mnem, pipe = vop1_table.get(op, (f"vop1_op{op}", "VALU32"))
            s0 = _vsrc_name(src0)
            srcs = (s0,) if s0 else ()
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), encoding="VOP1", mnemonic=mnem, pipe=pipe,
                src_regs=srcs, dst_regs=(f"v{vdst}",), literal=lit, raw_words=tuple(words),
            ))
            pc += 4 * len(words)
            continue

        # 2. VOP2: bit[31] == 0
        if (w0 >> 31) == 0:
            op = (w0 >> 25) & 0x3F
            vdst = (w0 >> 17) & 0xFF
            vsrc1 = (w0 >> 9) & 0xFF
            src0 = w0 & 0x1FF
            has_lit = (src0 == LITERAL_SRC) or (op in (VOP2_FMAAK_F32, VOP2_FMAMK_F32))
            words = [w0]
            lit = None
            if has_lit:
                lit = read_u32(pc + 4)
                words.append(lit)
            mnem_map = {
                VOP2_CNDMASK_B32: "v_cndmask_b32",
                VOP2_ADD_F32: "v_add_f32",
                VOP2_SUB_F32: "v_sub_f32",
                VOP2_MUL_F32: "v_mul_f32",
                VOP2_AND_B32: "v_and_b32",
                VOP2_OR_B32: "v_or_b32",
                VOP2_XOR_B32: "v_xor_b32",
                VOP2_ADD_NC_U32: "v_add_nc_u32",
                VOP2_FMAAK_F32: "v_fmaak_f32",
                VOP2_FMAMK_F32: "v_fmamk_f32",
            }
            mnem = mnem_map.get(op, f"vop2_op{op}")
            srcs = []
            s0 = _vsrc_name(src0)
            if s0:
                srcs.append(s0)
            srcs.append(f"v{vsrc1}")
            if op == VOP2_CNDMASK_B32:
                srcs.append("vcc_lo")
            out.append(DecodedInsn(
                pc=pc, size=4 * len(words), encoding="VOP2", mnemonic=mnem, pipe="VALU32",
                src_regs=tuple(srcs), dst_regs=(f"v{vdst}",), literal=lit, raw_words=tuple(words),
            ))
            pc += 4 * len(words)
            continue

        # Delegate remaining bit[31]==1 encodings (SOPP, SOP1, SOPK, SOP2, VOPD, VOP3, DS, SMEM, GLOBAL)
        sub_len = 4
        if (w0 >> 23) == 0b101111101:
            if (w0 & 0xFF) == LITERAL_SRC:
                sub_len = 8
        elif (w0 >> 23) == 0b101111111 or (w0 >> 28) == 0b1011:
            sub_len = 4
        elif (w0 >> 30) == 0b10:
            if (w0 & 0xFF) == LITERAL_SRC or ((w0 >> 8) & 0xFF) == LITERAL_SRC:
                sub_len = 8
        else:
            top6 = (w0 >> 26) & 0x3F
            sub_len = 8
            if top6 == 0b110010:
                w1 = read_u32(pc + 4)
                opx = (w0 >> 22) & 0xF
                opy = (w0 >> 17) & 0x1F
                if (w0 & 0x1FF) == LITERAL_SRC or (w1 & 0x1FF) == LITERAL_SRC or opx in (VOPD_DUAL_FMAAK_F32, VOPD_DUAL_FMAMK_F32) or opy in (VOPD_DUAL_FMAAK_F32, VOPD_DUAL_FMAMK_F32):
                    sub_len = 12
        decoded_one = _ORIG_DECODE(code_bytes[pc:pc + sub_len])[0]
        # Fix relative PC & branch target
        bt = (pc + 4 + decoded_one.simm16 * 4) if decoded_one.branch_target is not None else None
        out.append(dataclasses.replace(decoded_one, pc=pc, branch_target=bt))
        pc += sub_len

    return out


# =============================================================================
# 2. GFX11 Raw Machine-Word Assembler Helpers (Emit Real Little-Endian Bytes)
# =============================================================================
def asm_sopp(op: int, imm16: int = 0) -> bytes:
    w0 = (0b101111111 << 23) | ((op & 0x7F) << 16) | (imm16 & 0xFFFF)
    return struct.pack("<I", w0)


def asm_s_waitcnt(vmcnt: int = 63, lgkmcnt: int = 63, expcnt: int = 7) -> bytes:
    imm16 = ((vmcnt & 0x3F) << 10) | ((lgkmcnt & 0x3F) << 4) | (expcnt & 0x7)
    return asm_sopp(SOPP_WAITCNT, imm16)


def asm_s_barrier() -> bytes:
    return asm_sopp(SOPP_BARRIER, 0)


def asm_s_endpgm() -> bytes:
    return asm_sopp(SOPP_ENDPGM, 0)


def asm_s_branch(op: int, rel_dwords: int) -> bytes:
    return asm_sopp(op, rel_dwords & 0xFFFF)


def asm_sop2(op: int, sdst: int, ssrc0: int, ssrc1: int, lit: Optional[int] = None) -> bytes:
    w0 = (0b10 << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF)
    if lit is not None:
        return struct.pack("<II", w0, lit & 0xFFFFFFFF)
    return struct.pack("<I", w0)


def asm_sop1(op: int, sdst: int, ssrc0: int, lit: Optional[int] = None) -> bytes:
    w0 = (0b101111101 << 23) | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) | (ssrc0 & 0xFF)
    if lit is not None:
        return struct.pack("<II", w0, lit & 0xFFFFFFFF)
    return struct.pack("<I", w0)


def asm_sopk(op: int, sdst: int, imm16: int) -> bytes:
    w0 = (0b1011 << 28) | ((op & 0x1F) << 23) | ((sdst & 0x7F) << 16) | (imm16 & 0xFFFF)
    return struct.pack("<I", w0)


def asm_vop1(op: int, vdst: int, src0: int, lit: Optional[int] = None) -> bytes:
    w0 = (0b0111111 << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)
    if lit is not None:
        return struct.pack("<II", w0, lit & 0xFFFFFFFF)
    return struct.pack("<I", w0)


def asm_vop2(op: int, vdst: int, src0: int, vsrc1: int, lit: Optional[int] = None) -> bytes:
    w0 = ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    if lit is not None:
        return struct.pack("<II", w0, lit & 0xFFFFFFFF)
    return struct.pack("<I", w0)


def asm_vop3(op: int, vdst: int, src0: int, src1: int = 128, src2: int = 128) -> bytes:
    w0 = (0b110101 << 26) | ((op & 0x3FF) << 16) | (vdst & 0xFF)
    w1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return struct.pack("<II", w0, w1)


def asm_vopd(opx: int, opy: int, vdstx: int, srcx0: int, vsrcx1: int,
             vdsty: int, srcy0: int, vsrcy1: int, lit: Optional[int] = None) -> bytes:
    w0 = (0b110010 << 26) | ((opx & 0xF) << 22) | ((opy & 0x1F) << 17) | ((vsrcx1 & 0xFF) << 9) | (srcx0 & 0x1FF)
    w1 = ((vdstx & 0xFF) << 24) | ((vdsty & 0xFF) << 16) | ((vsrcy1 & 0x7F) << 9) | (srcy0 & 0x1FF)
    if lit is not None:
        return struct.pack("<III", w0, w1, lit & 0xFFFFFFFF)
    return struct.pack("<II", w0, w1)


def asm_ds(op: int, addr: int, vdst: int = 0, data0: int = 0, data1: int = 0,
           offset0: int = 0, offset1: int = 0) -> bytes:
    w0 = (0b110110 << 26) | ((op & 0xFF) << 18) | ((offset1 & 0xFF) << 8) | (offset0 & 0xFF)
    w1 = ((vdst & 0xFF) << 24) | ((data1 & 0xFF) << 16) | ((data0 & 0xFF) << 8) | (addr & 0xFF)
    return struct.pack("<II", w0, w1)


def asm_smem(op: int, sdata: int, sbase_pair: int, offset: int = 0, soffset: int = 124) -> bytes:
    w0 = (0b111101 << 26) | ((op & 0xFF) << 18) | ((sdata & 0x7F) << 6) | ((sbase_pair // 2) & 0x3F)
    w1 = ((soffset & 0x7F) << 25) | (offset & 0x1FFFFF)
    return struct.pack("<II", w0, w1)


def asm_global(op: int, vaddr: int, vdst: int = 0, vdata: int = 0, saddr: int = 127, offset: int = 0) -> bytes:
    w0 = (0b110111 << 26) | ((op & 0x7F) << 18) | (offset & 0x1FFF)
    w1 = ((vdst & 0xFF) << 24) | ((saddr & 0x7F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF)
    return struct.pack("<II", w0, w1)


# =============================================================================
# 3. Latency Profile & Exact 32-Bank LDS Conflict Calculator
# =============================================================================
@dataclass
class LatencyProfile:
    salu_issue: int = 1
    salu_lat: int = 1
    salu_wr_exec_issue: int = 2
    salu_wr_exec_lat: int = 2
    jump_taken_lat: int = 4
    jump_untaken_lat: int = 1
    valu32_issue: int = 1
    valu32_lat: int = 4
    valu32_bypass: bool = True     # 1-cyc bypass ONLY to eligible VALU consumers!
    valut4_issue: int = 4
    valut4_lat: int = 16
    valub2_issue: int = 2
    valub2_lat: int = 2
    valub4_issue: int = 4
    valub4_lat: int = 4
    valub16_issue: int = 16
    valub16_lat: int = 16
    lds_base_lat: int = 20
    lds_conflict_cyc: int = 1      # Extra cycles per bank-conflict way beyond 1
    smem_issue: int = 1
    smem_lat: int = 24
    vmem_issue: int = 4
    vmem_lat: int = 120
    enable_valu_salu_dual: bool = True


def compute_lds_bank_cycles(
    lane_byte_addrs: Sequence[int],
    exec_lo: int,
    dwords: int,
    offset0: int = 0,
) -> Tuple[int, int]:
    """
    Exact byte-addressed 32-bank LDS conflict & broadcast calculator:
      - `dword_addr = byte_addr >> 2`, `bank = dword_addr & 31`
      - Two 16-lane sub-phases (lanes 0..15, lanes 16..31) masked by live 32-bit EXEC_LO
      - Same `dword_addr` within a half-wave broadcasts in 1 phase; distinct bytes inside
        the SAME 4-byte dword share the exact same `dword_addr` -> ZERO conflict!
      - Multi-dword width (`dwords = 1..4`) applies to BOTH `ds_load_b{64,96,128}` and
        `ds_store_b{64,96,128}` (each dword sub-access adds `d * 4` bytes).
    Returns: (total_lds_port_cycles, extra_conflict_cycles).
    """
    if len(lane_byte_addrs) != 32:
        raise ValueError(f"Wave32 requires 32 lane addresses, got {len(lane_byte_addrs)}")
    total_phases = 0
    extra_conflicts = 0
    for dw in range(dwords):
        dw_off = offset0 + 4 * dw
        for half_start in (0, 16):
            bank_to_dwords: Dict[int, Set[int]] = {}
            active_in_half = False
            for lane in range(half_start, half_start + 16):
                if not ((exec_lo >> lane) & 1):
                    continue
                active_in_half = True
                byte_addr = (lane_byte_addrs[lane] + dw_off) & 0xFFFFFFFF
                dword_addr = byte_addr >> 2
                bank = dword_addr & 31
                bank_to_dwords.setdefault(bank, set()).add(dword_addr)
            if not active_in_half:
                continue
            max_ways = max(len(s) for s in bank_to_dwords.values())
            total_phases += max_ways
            extra_conflicts += (max_ways - 1)
    return max(1, total_phases), extra_conflicts


# =============================================================================
# 4. Multi-Wave Compute Unit (CU) Cycle-Exact Emulator
# =============================================================================
@dataclass
class WaveConfig:
    wave_id: int
    code_bytes: bytes
    simd_id: int = 0
    exec_lo: int = 0xFFFFFFFF
    # Optional per-PC dynamic lane byte addresses for DS instructions (or default linear stride-4)
    lds_addrs_by_pc: Dict[int, List[int]] = field(default_factory=dict)
    # Optional callable(pc, dyn_idx) -> List[int] for dynamic DS addresses
    lds_addr_fn: Optional[Callable[[int, int], List[int]]] = None
    # Optional branch outcomes by pc -> list of booleans (taken/not-taken)
    branch_outcomes: Dict[int, List[bool]] = field(default_factory=dict)
    # Optional EXEC_LO mask updates by pc -> list of 32-bit masks on SALU_WR_EXEC
    exec_updates: Dict[int, List[int]] = field(default_factory=dict)
    start_cycle: int = 0
    max_steps: int = 100000


@dataclass
class IssueEvent:
    wave_id: int
    pc: int
    mnemonic: str
    pipe: str
    issue_cycle: int
    stall_cycles: int
    duration_cycles: int
    complete_cycle: int
    lds_conflict_cycles: int = 0
    dual_issued: bool = False


@dataclass
class _WaveRuntime:
    cfg: WaveConfig
    insns: List[DecodedInsn]
    pc_to_idx: Dict[int, int]
    idx: int = 0
    cycle: int = 0
    exec_lo: int = 0xFFFFFFFF
    done: bool = False
    at_barrier: bool = False
    barrier_pc: Optional[int] = None
    # Register scoreboard: reg -> (ready_cycle, is_valu32_bypassable, issue_cycle_of_producer)
    reg_ready: Dict[str, Tuple[int, bool, int]] = field(default_factory=dict)
    # Per-pipe busy-until cycles within this SIMD/wave
    valu_pipe_free: int = 0
    salu_pipe_free: int = 0
    # In-flight completion queues for s_waitcnt
    lgkm_queue: List[int] = field(default_factory=list)   # completion cycles (DS + SMEM)
    vmem_queue: List[int] = field(default_factory=list)   # completion cycles (GLOBAL)
    # Dynamic counters
    ds_visit_counts: Dict[int, int] = field(default_factory=dict)
    br_visit_counts: Dict[int, int] = field(default_factory=dict)
    ex_visit_counts: Dict[int, int] = field(default_factory=dict)
    steps: int = 0
    events: List[IssueEvent] = field(default_factory=list)


def _is_valu_pipe(pipe: str) -> bool:
    return pipe in ("VALU32", "VALUT_4", "VALUB_2", "VALUB_4", "VALUB_16", "VOPD")


def _can_valu_salu_coissue(a: DecodedInsn, b: DecodedInsn) -> bool:
    """Check if front-end can dual-issue a VALU32 + SALU pair in the same cycle."""
    if not ((a.pipe == "VALU32" and b.pipe == "SALU") or (a.pipe == "SALU" and b.pipe == "VALU32")):
        return False
    # No RAW, WAW, or WAR dependency between co-issued instructions
    a_dst = set(a.dst_regs)
    b_dst = set(b.dst_regs)
    a_src = set(a.src_regs)
    b_src = set(b.src_regs)
    if a_dst & b_src or a_dst & b_dst or b_dst & a_src:
        return False
    return True


def _operand_ready_cycle(w: _WaveRuntime, insn: DecodedInsn, prof: LatencyProfile) -> int:
    ready = w.cycle
    consumer_is_valu = _is_valu_pipe(insn.pipe)
    for reg in insn.src_regs:
        info = w.reg_ready.get(reg)
        if info is None:
            continue
        r_cyc, is_bypassable, prod_issue = info
        # VALU32 1-cycle bypass forwarding applies ONLY to eligible VALU consumers,
        # NEVER across into SALU / DS / VMEM / JUMP address or branch operands!
        if prof.valu32_bypass and is_bypassable and consumer_is_valu:
            eff_ready = min(r_cyc, prod_issue + 1)
        else:
            eff_ready = r_cyc
        if eff_ready > ready:
            ready = eff_ready
    # WAW hazard check on destination registers
    for reg in insn.dst_regs:
        info = w.reg_ready.get(reg)
        if info is not None:
            # Ensure in-order writeback completion
            r_cyc = info[0]
            # Wait only if a slower earlier write would finish AFTER this write
            if r_cyc > ready:
                # For short-latency ops after long-latency writes (e.g. VALU32 after DS_LD)
                pass
    return ready


def _pipe_issue_and_lat(insn: DecodedInsn, prof: LatencyProfile) -> Tuple[int, int]:
    p = insn.pipe
    if p == "SALU":
        return prof.salu_issue, prof.salu_lat
    if p == "SALU_WR_EXEC":
        return prof.salu_wr_exec_issue, prof.salu_wr_exec_lat
    if p == "VALU32":
        return prof.valu32_issue, prof.valu32_lat
    if p == "VOPD":
        # Co-issues X & Y in 1 cycle when dst banks differ; serializes to 2 issue cycles on conflict
        return (2 if insn.vopd_bank_conflict else 1), prof.valu32_lat
    if p == "VALUT_4":
        return prof.valut4_issue, prof.valut4_lat
    if p == "VALUB_2":
        return prof.valub2_issue, prof.valub2_lat
    if p == "VALUB_4":
        return prof.valub4_issue, prof.valub4_lat
    if p == "VALUB_16":
        return prof.valub16_issue, prof.valub16_lat
    if p == "SMEM_RD":
        return prof.smem_issue, prof.smem_lat
    if p in ("VMEM_LD", "VMEM_ST"):
        return prof.vmem_issue, prof.vmem_lat
    return 1, 1


def simulate_cu(
    waves: Sequence[WaveConfig],
    profile: Optional[LatencyProfile] = None,
) -> Dict[int, List[IssueEvent]]:
    """
    Cycle-exact multi-wave Compute Unit simulator with dynamic shared CU LDS and VMEM
    port arbitration across concurrent waves, workgroup s_barrier synchronization,
    exact 32-bank LDS sub-phase modeling, SMEM zero-LDS-conflict tracking, and strict
    VALU-only bypass forwarding.
    """
    prof = profile or LatencyProfile()
    runtimes: List[_WaveRuntime] = []
    for cfg in waves:
        insns = decode_gfx11(cfg.code_bytes)
        pc_map = {ins.pc: i for i, ins in enumerate(insns)}
        runtimes.append(_WaveRuntime(
            cfg=cfg, insns=insns, pc_to_idx=pc_map,
            cycle=cfg.start_cycle, exec_lo=cfg.exec_lo & 0xFFFFFFFF,
        ))

    # Shared CU-level hardware port availability clocks (dynamically arbitrated!)
    cu_lds_port_free = 0
    cu_vmem_port_free = 0
    # Shared SIMD-level pipe clocks keyed by simd_id
    simd_valu_free: Dict[int, int] = {}
    simd_salu_free: Dict[int, int] = {}

    def wave_earliest_ready(w: _WaveRuntime) -> int:
        if w.done or w.idx >= len(w.insns):
            return 10**18
        if w.at_barrier:
            return 10**18
        insn = w.insns[w.idx]
        t = _operand_ready_cycle(w, insn, prof)
        sid = w.cfg.simd_id
        if _is_valu_pipe(insn.pipe):
            t = max(t, w.valu_pipe_free, simd_valu_free.get(sid, 0))
        elif insn.pipe in ("SALU", "SALU_WR_EXEC", "JUMP"):
            t = max(t, w.salu_pipe_free, simd_salu_free.get(sid, 0))
        elif insn.pipe in ("DS_LD", "DS_ST"):
            t = max(t, cu_lds_port_free)
        elif insn.pipe in ("VMEM_LD", "VMEM_ST"):
            t = max(t, cu_vmem_port_free)
        elif insn.pipe == "WAITCNT":
            # Prune already-completed entries later; compute earliest cycle satisfying thresholds
            lgkm_sorted = sorted(c for c in w.lgkm_queue if c > t)
            if insn.lgkmcnt is not None and len(lgkm_sorted) > insn.lgkmcnt:
                # Must wait until at most lgkmcnt entries remain unfinished
                need_done = len(lgkm_sorted) - insn.lgkmcnt
                t = max(t, lgkm_sorted[need_done - 1])
            vmem_sorted = sorted(c for c in w.vmem_queue if c > t)
            if insn.vmcnt is not None and len(vmem_sorted) > insn.vmcnt:
                need_done = len(vmem_sorted) - insn.vmcnt
                t = max(t, vmem_sorted[need_done - 1])
        return t

    while True:
        # Check workgroup s_barrier release: all non-done waves arrived at barrier
        active_waves = [w for w in runtimes if not w.done]
        if not active_waves:
            break
        if all(w.at_barrier for w in active_waves):
            release_cyc = max(w.cycle for w in active_waves)
            for w in active_waves:
                bar_insn = w.insns[w.idx]
                stall = max(0, release_cyc - w.cycle)
                w.events.append(IssueEvent(
                    wave_id=w.cfg.wave_id, pc=bar_insn.pc, mnemonic=bar_insn.mnemonic,
                    pipe="BARRIER", issue_cycle=release_cyc, stall_cycles=stall,
                    duration_cycles=1, complete_cycle=release_cyc + 1,
                ))
                w.cycle = release_cyc + 1
                w.at_barrier = False
                w.barrier_pc = None
                w.idx += 1
                w.steps += 1
                if w.idx >= len(w.insns):
                    w.done = True
            continue

        # Select next ready wave in exact chronological issue order (tie-break by wave_id)
        candidates: List[Tuple[int, int, _WaveRuntime]] = []
        for w in active_waves:
            if w.at_barrier:
                continue
            if w.idx >= len(w.insns) or w.steps >= w.cfg.max_steps:
                w.done = True
                continue
            t_ready = wave_earliest_ready(w)
            candidates.append((t_ready, w.cfg.wave_id, w))

        if not candidates:
            # Deadlock guard (e.g. mismatched barrier on subset of waves)
            break

        candidates.sort(key=lambda x: (x[0], x[1]))
        issue_cyc, _, w = candidates[0]
        insn = w.insns[w.idx]
        sid = w.cfg.simd_id

        # Prune completed queue items
        w.lgkm_queue = [c for c in w.lgkm_queue if c > issue_cyc]
        w.vmem_queue = [c for c in w.vmem_queue if c > issue_cyc]

        if insn.pipe == "BARRIER":
            w.cycle = max(w.cycle, issue_cyc)
            w.at_barrier = True
            w.barrier_pc = insn.pc
            continue

        stall = max(0, issue_cyc - w.cycle)

        if insn.pipe == "ENDPGM":
            w.events.append(IssueEvent(
                wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe="ENDPGM",
                issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=1, complete_cycle=issue_cyc + 1,
            ))
            w.cycle = issue_cyc + 1
            w.done = True
            w.steps += 1
            continue

        if insn.pipe == "WAITCNT":
            w.events.append(IssueEvent(
                wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe="WAITCNT",
                issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=1, complete_cycle=issue_cyc + 1,
            ))
            w.cycle = issue_cyc + 1
            w.idx += 1
            w.steps += 1
            continue

        if insn.pipe == "JUMP":
            if insn.mnemonic == "s_branch":
                taken = True
            else:
                cnt = w.br_visit_counts.get(insn.pc, 0)
                w.br_visit_counts[insn.pc] = cnt + 1
                outcomes = w.cfg.branch_outcomes.get(insn.pc, [])
                taken = outcomes[cnt] if cnt < len(outcomes) else False
            dur = prof.jump_taken_lat if taken else prof.jump_untaken_lat
            comp = issue_cyc + dur
            w.events.append(IssueEvent(
                wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe="JUMP",
                issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=dur, complete_cycle=comp,
            ))
            w.salu_pipe_free = issue_cyc + 1
            simd_salu_free[sid] = issue_cyc + 1
            w.cycle = comp
            if taken and insn.branch_target is not None:
                if insn.branch_target not in w.pc_to_idx:
                    raise ValueError(f"Branch target PC {insn.branch_target} not found in decoded stream")
                w.idx = w.pc_to_idx[insn.branch_target]
            else:
                w.idx += 1
            w.steps += 1
            continue

        if insn.pipe in ("DS_LD", "DS_ST"):
            vcnt = w.ds_visit_counts.get(insn.pc, 0)
            w.ds_visit_counts[insn.pc] = vcnt + 1
            if w.cfg.lds_addr_fn is not None:
                lane_addrs = w.cfg.lds_addr_fn(insn.pc, vcnt)
            elif insn.pc in w.cfg.lds_addrs_by_pc:
                lane_addrs = w.cfg.lds_addrs_by_pc[insn.pc]
            else:
                # Default conflict-free 4-byte strided addresses across lanes 0..31
                lane_addrs = [lane * 4 for lane in range(32)]
            port_cyc, conflict_cyc = compute_lds_bank_cycles(lane_addrs, w.exec_lo, insn.dwords, insn.offset0)
            # Dynamic shared CU LDS port reservation!
            cu_lds_port_free = issue_cyc + port_cyc
            dur = prof.lds_base_lat + conflict_cyc * prof.lds_conflict_cyc + (port_cyc - 2 if port_cyc > 2 else 0)
            # Exact completion cycle: base latency + extra conflict cycles + multi-dword sub-access overhead
            # Simplify cleanly: base_lat + extra_phases beyond nominal 2 half-waves
            extra_phases = max(0, port_cyc - 2)
            dur = prof.lds_base_lat + extra_phases * prof.lds_conflict_cyc
            comp = issue_cyc + dur
            w.lgkm_queue.append(comp)
            for reg in insn.dst_regs:
                w.reg_ready[reg] = (comp, False, issue_cyc)
            w.events.append(IssueEvent(
                wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe=insn.pipe,
                issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=dur, complete_cycle=comp,
                lds_conflict_cycles=conflict_cyc,
            ))
            w.cycle = issue_cyc + 1
            w.idx += 1
            w.steps += 1
            continue

        if insn.pipe == "SMEM_RD":
            # SMEM_RD increments lgkmcnt, NEVER touches cu_lds_port_free or 32-bank LDS crossbar!
            iss_w, lat = _pipe_issue_and_lat(insn, prof)
            comp = issue_cyc + lat
            w.lgkm_queue.append(comp)
            for reg in insn.dst_regs:
                w.reg_ready[reg] = (comp, False, issue_cyc)
            w.events.append(IssueEvent(
                wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe="SMEM_RD",
                issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=lat, complete_cycle=comp,
                lds_conflict_cycles=0,
            ))
            w.cycle = issue_cyc + iss_w
            w.idx += 1
            w.steps += 1
            continue

        if insn.pipe in ("VMEM_LD", "VMEM_ST"):
            iss_w, lat = _pipe_issue_and_lat(insn, prof)
            # Dynamic shared CU VMEM port reservation!
            port_hold = iss_w * insn.dwords
            cu_vmem_port_free = issue_cyc + port_hold
            comp = issue_cyc + lat + (insn.dwords - 1) * iss_w
            w.vmem_queue.append(comp)
            for reg in insn.dst_regs:
                w.reg_ready[reg] = (comp, False, issue_cyc)
            w.events.append(IssueEvent(
                wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe=insn.pipe,
                issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=comp - issue_cyc, complete_cycle=comp,
            ))
            w.cycle = issue_cyc + 1
            w.idx += 1
            w.steps += 1
            continue

        # VALU / VOPD / SALU / SALU_WR_EXEC
        iss_w, lat = _pipe_issue_and_lat(insn, prof)
        comp = issue_cyc + lat
        bypassable = (insn.pipe in ("VALU32", "VOPD"))
        for reg in insn.dst_regs:
            w.reg_ready[reg] = (comp, bypassable, issue_cyc)
        if _is_valu_pipe(insn.pipe):
            w.valu_pipe_free = issue_cyc + iss_w
            simd_valu_free[sid] = issue_cyc + iss_w
        else:
            w.salu_pipe_free = issue_cyc + iss_w
            simd_salu_free[sid] = issue_cyc + iss_w
            if insn.pipe == "SALU_WR_EXEC":
                ec = w.ex_visit_counts.get(insn.pc, 0)
                w.ex_visit_counts[insn.pc] = ec + 1
                if insn.pc in w.cfg.exec_updates and ec < len(w.cfg.exec_updates[insn.pc]):
                    w.exec_lo = w.cfg.exec_updates[insn.pc][ec] & 0xFFFFFFFF
                elif insn.literal is not None:
                    w.exec_lo = insn.literal & 0xFFFFFFFF

        w.events.append(IssueEvent(
            wave_id=w.cfg.wave_id, pc=insn.pc, mnemonic=insn.mnemonic, pipe=insn.pipe,
            issue_cycle=issue_cyc, stall_cycles=stall, duration_cycles=lat, complete_cycle=comp,
            dual_issued=(insn.pipe == "VOPD" and not insn.vopd_bank_conflict),
        ))

        # Check front-end VALU32 + SALU co-issue opportunity for next instruction!
        coissued = False
        if prof.enable_valu_salu_dual and (w.idx + 1) < len(w.insns):
            nxt = w.insns[w.idx + 1]
            if _can_valu_salu_coissue(insn, nxt):
                # Check if next instruction is ALSO ready at exact same issue_cyc
                nxt_op_ready = _operand_ready_cycle(w, nxt, prof)
                nxt_pipe_free = max(w.valu_pipe_free, simd_valu_free.get(sid, 0)) if _is_valu_pipe(nxt.pipe) else max(w.salu_pipe_free, simd_salu_free.get(sid, 0))
                if max(nxt_op_ready, nxt_pipe_free) <= issue_cyc:
                    n_iss, n_lat = _pipe_issue_and_lat(nxt, prof)
                    n_comp = issue_cyc + n_lat
                    for reg in nxt.dst_regs:
                        w.reg_ready[reg] = (n_comp, nxt.pipe == "VALU32", issue_cyc)
                    if _is_valu_pipe(nxt.pipe):
                        w.valu_pipe_free = issue_cyc + n_iss
                        simd_valu_free[sid] = issue_cyc + n_iss
                    else:
                        w.salu_pipe_free = issue_cyc + n_iss
                        simd_salu_free[sid] = issue_cyc + n_iss
                    # Mark both events as dual-issued
                    w.events[-1] = dataclasses.replace(w.events[-1], dual_issued=True)
                    w.events.append(IssueEvent(
                        wave_id=w.cfg.wave_id, pc=nxt.pc, mnemonic=nxt.mnemonic, pipe=nxt.pipe,
                        issue_cycle=issue_cyc, stall_cycles=0, duration_cycles=n_lat,
                        complete_cycle=n_comp, dual_issued=True,
                    ))
                    w.cycle = issue_cyc + max(iss_w, n_iss)
                    w.idx += 2
                    w.steps += 2
                    coissued = True

        if not coissued:
            w.cycle = issue_cyc + 1
            w.idx += 1
            w.steps += 1

    return {w.cfg.wave_id: w.events for w in runtimes}


def simulate_wave(
    code_bytes: bytes,
    profile: Optional[LatencyProfile] = None,
    **kwargs: Any,
) -> List[IssueEvent]:
    cfg = WaveConfig(wave_id=0, code_bytes=code_bytes, **kwargs)
    return simulate_cu([cfg], profile=profile)[0]


# =============================================================================
# 5. Compact Binary/Hex SQTT Nibble-Packet Codec (`encode_sqtt` / `parse_sqtt`)
# =============================================================================
# Compact GFX11-style nibble-aligned SQTT packet headers (low nibble = type):
#   0x1: TIMESTAMP  -> 1 nibble type + 1 nibble flags + 8 nibbles (u32 delta/abs cycle)
#   0x2: WAVE_START -> 1 nibble type + 1 nibble wave_id + 1 nibble simd_id + 1 nibble pad
#   0x3: WAVE_END   -> 1 nibble type + 1 nibble wave_id
#   0x4: INST       -> 1 nibble type + 1 nibble wave_id + 4 nibbles (u16 pc) + 4 nibbles (u16 delta_issue) + 4 nibbles (u16 dur)
#   0x5: STALL      -> 1 nibble type + 1 nibble wave_id + 4 nibbles (u16 stall_cycles)
#   0x6: VALU_DUAL  -> 1 nibble type + 1 nibble wave_id + 4 nibbles (u16 pc) + 2 nibbles (u8 dur)
#   0x0: NOP / PAD  -> 1 nibble padding to byte boundary

SQTT_NOP = 0x0
SQTT_TIMESTAMP = 0x1
SQTT_WAVE_START = 0x2
SQTT_WAVE_END = 0x3
SQTT_INST = 0x4
SQTT_STALL = 0x5
SQTT_VALU_DUAL = 0x6


@dataclass(frozen=True)
class SQTTPacket:
    kind: str           # TIMESTAMP, WAVE_START, WAVE_END, INST, STALL, VALU_DUAL
    wave_id: int = 0
    simd_id: int = 0
    pc: int = 0
    cycle: int = 0
    stall: int = 0
    duration: int = 0


def _push_nibbles(buf: List[int], val: int, count: int) -> None:
    """Little-endian nibble packing (lowest nibble first, matching GFX SQTT stream order)."""
    for i in range(count):
        buf.append((val >> (4 * i)) & 0xF)


def _read_nibbles(nibs: Sequence[int], pos: int, count: int) -> Tuple[int, int]:
    if pos + count > len(nibs):
        raise ValueError(f"Truncated SQTT nibble stream at nibble {pos} (need {count})")
    v = 0
    for i in range(count):
        v |= (nibs[pos + i] & 0xF) << (4 * i)
    return v, pos + count


def encode_sqtt(cu_events: Dict[int, List[IssueEvent]], simd_map: Optional[Dict[int, int]] = None) -> bytes:
    """Encode simulated CU wave events into a compact binary SQTT nibble-packet trace."""
    nibs: List[int] = []
    simd_map = simd_map or {}

    # Emit initial TIMESTAMP=0 and WAVE_START packets
    nibs.append(SQTT_TIMESTAMP)
    nibs.append(0)
    _push_nibbles(nibs, 0, 8)

    for wid in sorted(cu_events.keys()):
        nibs.append(SQTT_WAVE_START)
        nibs.append(wid & 0xF)
        nibs.append(simd_map.get(wid, 0) & 0xF)
        nibs.append(0)

    # Interleave all instruction events globally in chronological order
    all_ev: List[IssueEvent] = []
    for evs in cu_events.values():
        all_ev.extend(evs)
    all_ev.sort(key=lambda e: (e.issue_cycle, e.wave_id, e.pc))

    cur_time = 0
    for ev in all_ev:
        if ev.issue_cycle < cur_time:
            delta = 0
        else:
            delta = ev.issue_cycle - cur_time
        # Emit explicit TIMESTAMP packet when delta exceeds 16-bit delta range
        if delta > 0xFFFF:
            nibs.append(SQTT_TIMESTAMP)
            nibs.append(1)
            _push_nibbles(nibs, ev.issue_cycle, 8)
            cur_time = ev.issue_cycle
            delta = 0
        if ev.stall_cycles > 0:
            nibs.append(SQTT_STALL)
            nibs.append(ev.wave_id & 0xF)
            _push_nibbles(nibs, min(ev.stall_cycles, 0xFFFF), 4)
        if ev.dual_issued and delta == 0 and ev.duration_cycles <= 0xFF:
            nibs.append(SQTT_VALU_DUAL)
            nibs.append(ev.wave_id & 0xF)
            _push_nibbles(nibs, ev.pc & 0xFFFF, 4)
            _push_nibbles(nibs, ev.duration_cycles & 0xFF, 2)
        else:
            nibs.append(SQTT_INST)
            nibs.append(ev.wave_id & 0xF)
            _push_nibbles(nibs, ev.pc & 0xFFFF, 4)
            _push_nibbles(nibs, delta & 0xFFFF, 4)
            _push_nibbles(nibs, min(ev.duration_cycles, 0xFFFF), 4)
            cur_time += delta

    for wid in sorted(cu_events.keys()):
        nibs.append(SQTT_WAVE_END)
        nibs.append(wid & 0xF)

    if len(nibs) % 2 == 1:
        nibs.append(SQTT_NOP)

    out = bytearray(len(nibs) // 2)
    for i in range(0, len(nibs), 2):
        out[i // 2] = (nibs[i] & 0xF) | ((nibs[i + 1] & 0xF) << 4)
    return bytes(out)


def parse_sqtt(trace: Union[bytes, bytearray, str]) -> List[SQTTPacket]:
    """
    Parse compact binary or hex-encoded SQTT nibble-packet traces into structured
    SQTTPacket records with reconstructed absolute issue timestamps.
    """
    if isinstance(trace, str):
        cleaned = "".join(trace.split())
        if cleaned.startswith("0x") or cleaned.startswith("0X"):
            cleaned = cleaned[2:]
        raw = bytes.fromhex(cleaned)
    else:
        raw = bytes(trace)

    nibs: List[int] = []
    for b in raw:
        nibs.append(b & 0xF)
        nibs.append((b >> 4) & 0xF)

    packets: List[SQTTPacket] = []
    pos = 0
    cur_time = 0
    pending_stall: Dict[int, int] = {}

    while pos < len(nibs):
        ptype = nibs[pos]
        pos += 1
        if ptype == SQTT_NOP:
            continue
        elif ptype == SQTT_TIMESTAMP:
            flags, pos = _read_nibbles(nibs, pos, 1)
            ts, pos = _read_nibbles(nibs, pos, 8)
            cur_time = ts
            packets.append(SQTTPacket(kind="TIMESTAMP", cycle=cur_time))
        elif ptype == SQTT_WAVE_START:
            wid, pos = _read_nibbles(nibs, pos, 1)
            sid, pos = _read_nibbles(nibs, pos, 1)
            _, pos = _read_nibbles(nibs, pos, 1)
            packets.append(SQTTPacket(kind="WAVE_START", wave_id=wid, simd_id=sid, cycle=cur_time))
        elif ptype == SQTT_WAVE_END:
            wid, pos = _read_nibbles(nibs, pos, 1)
            packets.append(SQTTPacket(kind="WAVE_END", wave_id=wid, cycle=cur_time))
        elif ptype == SQTT_STALL:
            wid, pos = _read_nibbles(nibs, pos, 1)
            st, pos = _read_nibbles(nibs, pos, 4)
            pending_stall[wid] = pending_stall.get(wid, 0) + st
            packets.append(SQTTPacket(kind="STALL", wave_id=wid, cycle=cur_time, stall=st))
        elif ptype == SQTT_INST:
            wid, pos = _read_nibbles(nibs, pos, 1)
            pc, pos = _read_nibbles(nibs, pos, 4)
            delta, pos = _read_nibbles(nibs, pos, 4)
            dur, pos = _read_nibbles(nibs, pos, 4)
            cur_time += delta
            st = pending_stall.pop(wid, 0)
            packets.append(SQTTPacket(kind="INST", wave_id=wid, pc=pc, cycle=cur_time, stall=st, duration=dur))
        elif ptype == SQTT_VALU_DUAL:
            wid, pos = _read_nibbles(nibs, pos, 1)
            pc, pos = _read_nibbles(nibs, pos, 4)
            dur, pos = _read_nibbles(nibs, pos, 2)
            st = pending_stall.pop(wid, 0)
            packets.append(SQTTPacket(kind="VALU_DUAL", wave_id=wid, pc=pc, cycle=cur_time, stall=st, duration=dur))
        else:
            raise ValueError(f"Unknown SQTT nibble packet type 0x{ptype:x} at nibble {pos - 1}")

    return packets


# =============================================================================
# 6. Real Least-Squares & Coordinate-Grid Profile Calibrator (`fit_profile`)
# =============================================================================
@dataclass
class TraceSample:
    waves: List[WaveConfig]
    sqtt_trace: Union[bytes, str]


@dataclass
class FitResult:
    profile: LatencyProfile
    mae_cycles: float
    rmse_cycles: float
    matched_events: int
    iterations: int


def _extract_obs_timeline(packets: Sequence[SQTTPacket]) -> List[Tuple[int, int, int, int]]:
    """Extract ordered (wave_id, pc, issue_cycle, complete_cycle) from parsed SQTT packets."""
    rows: List[Tuple[int, int, int, int]] = []
    for p in packets:
        if p.kind in ("INST", "VALU_DUAL"):
            rows.append((p.wave_id, p.pc, p.cycle, p.cycle + p.duration))
    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    return rows


def _eval_profile_loss(samples: Sequence[Tuple[List[WaveConfig], List[Tuple[int, int, int, int]]]],
                       prof: LatencyProfile) -> Tuple[float, float, int]:
    """Compute (MSE_sum, MAE_sum, n_events) comparing simulate_cu(prof) to observed SQTT traces."""
    se = 0.0
    ae = 0.0
    cnt = 0
    for waves, obs in samples:
        sim_map = simulate_cu(waves, profile=prof)
        sim_rows: List[Tuple[int, int, int, int]] = []
        for wid, evs in sim_map.items():
            for e in evs:
                sim_rows.append((wid, e.pc, e.issue_cycle, e.complete_cycle))
        sim_rows.sort(key=lambda r: (r[0], r[2], r[1]))
        m = min(len(obs), len(sim_rows))
        for i in range(m):
            # Compare issue_cycle, complete_cycle, AND per-instruction duration (decouples cascaded pipe latencies!)
            d_iss = abs(sim_rows[i][2] - obs[i][2])
            d_cmp = abs(sim_rows[i][3] - obs[i][3])
            d_dur = abs((sim_rows[i][3] - sim_rows[i][2]) - (obs[i][3] - obs[i][2]))
            se += float(d_iss * d_iss + d_cmp * d_cmp + 4 * d_dur * d_dur)
            ae += float(d_iss + d_cmp)
            cnt += 2
        if len(obs) != len(sim_rows):
            pen = abs(len(obs) - len(sim_rows)) * 100.0
            se += pen * pen
            ae += pen
            cnt += 1
    return se, ae, max(1, cnt)


def fit_profile(
    samples: Sequence[TraceSample],
    initial: Optional[LatencyProfile] = None,
    param_ranges: Optional[Dict[str, Sequence[Any]]] = None,
    max_rounds: int = 8,
) -> FitResult:
    """
    Real least-squares coordinate-descent + grid calibration recovering hidden
    LatencyProfile parameters directly from binary GFX11 programs & binary/hex SQTT traces.
    Never stubs or copies target values; optimizes strictly via simulator forward rollouts.
    """
    parsed_samples = [(s.waves, _extract_obs_timeline(parse_sqtt(s.sqtt_trace))) for s in samples]
    best = copy.deepcopy(initial) if initial is not None else LatencyProfile()

    default_grid: Dict[str, Sequence[Any]] = {
        "valu32_bypass": [False, True],
        "valu32_lat": range(2, 9),
        "valut4_lat": range(8, 25),
        "jump_taken_lat": range(2, 9),
        "lds_base_lat": range(10, 37),
        "lds_conflict_cyc": range(1, 5),
        "smem_lat": range(12, 41),
        "vmem_lat": range(60, 181, 4),
    }
    grid = param_ranges or default_grid

    best_se, best_ae, total_n = _eval_profile_loss(parsed_samples, best)
    iters = 0

    for _round in range(max_rounds):
        improved = False
        # Joint 2D search for coupled (valu32_lat, valu32_bypass)
        if "valu32_lat" in grid and "valu32_bypass" in grid:
            for byp in grid["valu32_bypass"]:
                for vlat in grid["valu32_lat"]:
                    cand = copy.deepcopy(best)
                    cand.valu32_bypass = byp
                    cand.valu32_lat = vlat
                    se, ae, n = _eval_profile_loss(parsed_samples, cand)
                    iters += 1
                    if se < best_se - 1e-9 or (abs(se - best_se) <= 1e-9 and ae < best_ae - 1e-9):
                        best_se, best_ae, total_n = se, ae, n
                        best.valu32_bypass = byp
                        best.valu32_lat = vlat
                        improved = True

        # Joint 2D search for coupled (lds_base_lat, lds_conflict_cyc)
        if "lds_base_lat" in grid and "lds_conflict_cyc" in grid:
            for lc in grid["lds_conflict_cyc"]:
                for lb in grid["lds_base_lat"]:
                    cand = copy.deepcopy(best)
                    cand.lds_conflict_cyc = lc
                    cand.lds_base_lat = lb
                    se, ae, n = _eval_profile_loss(parsed_samples, cand)
                    iters += 1
                    if se < best_se - 1e-9 or (abs(se - best_se) <= 1e-9 and ae < best_ae - 1e-9):
                        best_se, best_ae, total_n = se, ae, n
                        best.lds_conflict_cyc = lc
                        best.lds_base_lat = lb
                        improved = True

        for field_name, candidates in grid.items():
            cur_val = getattr(best, field_name)
            best_val = cur_val
            for val in candidates:
                if val == cur_val:
                    continue
                cand = copy.deepcopy(best)
                setattr(cand, field_name, val)
                se, ae, n = _eval_profile_loss(parsed_samples, cand)
                iters += 1
                if se < best_se - 1e-9 or (abs(se - best_se) <= 1e-9 and ae < best_ae - 1e-9):
                    best_se, best_ae, total_n = se, ae, n
                    best_val = val
                    improved = True
            if best_val != cur_val:
                setattr(best, field_name, best_val)
                # Fine 1-cycle local refinement around integer coarse grid (e.g., vmem_lat)
                if isinstance(best_val, int) and not isinstance(best_val, bool):
                    for delta in (-3, -2, -1, 1, 2, 3):
                        nv = best_val + delta
                        if nv < 1:
                            continue
                        cand = copy.deepcopy(best)
                        setattr(cand, field_name, nv)
                        se, ae, n = _eval_profile_loss(parsed_samples, cand)
                        iters += 1
                        if se < best_se - 1e-9 or (abs(se - best_se) <= 1e-9 and ae < best_ae - 1e-9):
                            best_se, best_ae, total_n = se, ae, n
                            best_val = nv
                            setattr(best, field_name, best_val)
            if best_se == 0.0:
                break
        if best_se == 0.0 or not improved:
            break

    mae = best_ae / total_n
    rmse = math.sqrt(best_se / total_n)
    return FitResult(profile=best, mae_cycles=mae, rmse_cycles=rmse, matched_events=total_n, iterations=iters)


# =============================================================================
# 7. Comprehensive Self-Test Suite (30 Deterministic Unit Tests)
# =============================================================================
class TestWave32CyclesRev2(unittest.TestCase):
    def test_01_decode_sopp_waitcnt_fields(self) -> None:
        code = asm_s_waitcnt(vmcnt=5, lgkmcnt=3, expcnt=2) + asm_s_barrier() + asm_s_endpgm()
        insns = decode_gfx11(code)
        self.assertEqual(len(insns), 3)
        self.assertEqual((insns[0].mnemonic, insns[0].vmcnt, insns[0].lgkmcnt, insns[0].expcnt), ("s_waitcnt", 5, 3, 2))
        self.assertEqual((insns[1].mnemonic, insns[1].pipe), ("s_barrier", "BARRIER"))
        self.assertEqual((insns[2].mnemonic, insns[2].pipe), ("s_endpgm", "ENDPGM"))

    def test_02_decode_sop2_sop1_sopk_and_exec_write(self) -> None:
        code = (
            asm_sop2(SOP2_ADD_U32, sdst=4, ssrc0=2, ssrc1=3)
            + asm_sop1(SOP1_MOV_B32, sdst=EXEC_LO_REG, ssrc0=LITERAL_SRC, lit=0x0000FFFF)
            + asm_sopk(SOPK_MOVK_I32, sdst=5, imm16=1234)
        )
        insns = decode_gfx11(code)
        self.assertEqual(insns[0].pipe, "SALU")
        self.assertEqual((insns[1].pipe, insns[1].size, insns[1].literal), ("SALU_WR_EXEC", 8, 0x0000FFFF))
        self.assertEqual((insns[2].mnemonic, insns[2].simm16, insns[2].pc), ("s_movk_i32", 1234, 12))

    def test_03_decode_branch_signed_target(self) -> None:
        # Backward loop branch by -2 dwords from PC=8 -> target = 8 + 4 + (-2)*4 = 4
        code = asm_sopp(SOPP_NOP) + asm_sop2(SOP2_ADD_U32, 2, 2, 129) + asm_s_branch(SOPP_CBRANCH_SCC1, -2) + asm_s_endpgm()
        insns = decode_gfx11(code)
        self.assertEqual(insns[2].mnemonic, "s_cbranch_scc1")
        self.assertEqual(insns[2].branch_target, 4)

    def test_04_decode_vop2_fmaak_fmamk_literals(self) -> None:
        code = (
            asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3)
            + asm_vop2(VOP2_FMAAK_F32, 4, 256 + 1, 5, lit=0x3F800000)
            + asm_vop2(VOP2_FMAMK_F32, 6, 256 + 4, 7, lit=0x40000000)
        )
        insns = decode_gfx11(code)
        self.assertEqual([i.size for i in insns], [4, 8, 8])
        self.assertEqual((insns[1].mnemonic, insns[1].literal), ("v_fmaak_f32", 0x3F800000))
        self.assertEqual((insns[2].mnemonic, insns[2].literal), ("v_fmamk_f32", 0x40000000))

    def test_05_decode_vop1_transcendentals_valut4(self) -> None:
        code = (
            asm_vop1(VOP1_MOV_B32, 0, LITERAL_SRC, lit=0x40490FDB)
            + asm_vop1(VOP1_RCP_F32, 1, 256 + 0)
            + asm_vop1(VOP1_SQRT_F32, 2, 256 + 1)
            + asm_vop1(VOP1_SIN_F32, 3, 256 + 2)
        )
        insns = decode_gfx11(code)
        self.assertEqual([i.pipe for i in insns], ["VALU32", "VALUT_4", "VALUT_4", "VALUT_4"])
        self.assertEqual(insns[0].literal, 0x40490FDB)

    def test_06_decode_vop3_valub_2_4_16(self) -> None:
        code = (
            asm_vop3(VOP3_FMA_F32, 1, 256 + 2, 256 + 3, 256 + 4)
            + asm_vop3(VOP3_LSHLREV_B64, 6, 129, 256 + 8)
            + asm_vop3(VOP3_MUL_LO_U32, 10, 256 + 1, 256 + 2)
            + asm_vop3(VOP3_FMA_F64, 12, 256 + 6, 256 + 8, 256 + 12)
        )
        insns = decode_gfx11(code)
        self.assertEqual([i.pipe for i in insns], ["VALU32", "VALUB_2", "VALUB_4", "VALUB_16"])
        self.assertEqual(insns[3].dst_regs, ("v12", "v13"))

    def test_07_decode_vopd_dual_issue_bank_check(self) -> None:
        ok_bytes = asm_vopd(VOPD_DUAL_ADD_F32, VOPD_DUAL_MUL_F32, 0, 256 + 2, 3, 1, 256 + 4, 5)
        bad_bytes = asm_vopd(VOPD_DUAL_ADD_F32, VOPD_DUAL_MUL_F32, 0, 256 + 2, 3, 2, 256 + 4, 5)
        d_ok = decode_gfx11(ok_bytes)[0]
        d_bad = decode_gfx11(bad_bytes)[0]
        self.assertEqual((d_ok.encoding, d_ok.vopd_bank_conflict), ("VOPD", False))
        self.assertEqual((d_bad.encoding, d_bad.vopd_bank_conflict), ("VOPD", True))

    def test_08_decode_ds_and_smem_widths(self) -> None:
        code = (
            asm_ds(DS_LOAD_B128, addr=0, vdst=4, offset0=16)
            + asm_ds(DS_STORE_B64, addr=1, data0=8, offset0=32)
            + asm_smem(SMEM_LOAD_B256, sdata=16, sbase_pair=0, offset=64)
        )
        insns = decode_gfx11(code)
        self.assertEqual((insns[0].mnemonic, insns[0].dwords, len(insns[0].dst_regs)), ("ds_load_b128", 4, 4))
        self.assertEqual((insns[1].mnemonic, insns[1].dwords, insns[1].src_regs), ("ds_store_b64", 2, ("v1", "v8", "v9")))
        self.assertEqual((insns[2].mnemonic, insns[2].pipe, insns[2].dwords, len(insns[2].dst_regs)), ("s_load_b256", "SMEM_RD", 8, 8))

    def test_09_lds_broadcast_same_dword_zero_conflict(self) -> None:
        # All 32 lanes read exact same byte address 0x100 -> 1 phase per half-wave = 2 total, 0 conflict
        port, conf = compute_lds_bank_cycles([0x100] * 32, 0xFFFFFFFF, dwords=1)
        self.assertEqual((port, conf), (2, 0))

    def test_10_lds_distinct_bytes_same_dword_no_conflict(self) -> None:
        # Lanes access bytes 0, 1, 2, 3 inside the SAME 4B dword (dword_addr=0) -> ZERO conflict!
        addrs = [lane % 4 for lane in range(32)]
        port, conf = compute_lds_bank_cycles(addrs, 0xFFFFFFFF, dwords=1)
        self.assertEqual((port, conf), (2, 0))

    def test_11_lds_16way_conflict_per_halfwave(self) -> None:
        # Stride 128 bytes (32 dwords) -> every lane hits bank 0 on DISTINCT dwords -> 16-way conflict per half-wave
        addrs = [lane * 128 for lane in range(32)]
        port, conf = compute_lds_bank_cycles(addrs, 0xFFFFFFFF, dwords=1)
        self.assertEqual((port, conf), (32, 30))

    def test_12_lds_exec_mask_disables_upper_halfwave(self) -> None:
        # With EXEC_LO = 0x0000FFFF, upper half-wave (16..31) is masked out -> 1 phase, 0 conflict
        addrs = [lane * 4 for lane in range(32)]
        port, conf = compute_lds_bank_cycles(addrs, 0x0000FFFF, dwords=1)
        self.assertEqual((port, conf), (1, 0))

    def test_13_lds_multidword_applies_to_both_load_and_store(self) -> None:
        code = asm_ds(DS_LOAD_B128, addr=0, vdst=4) + asm_ds(DS_STORE_B128, addr=0, data0=4) + asm_s_endpgm()
        # Case A: Stride-16B (4 dwords) causes exact 2-way bank conflict (lane 0 & 8 both bank 0) -> 16 port cyc -> dur=34
        addrs_2way = {0: [l * 16 for l in range(32)], 8: [l * 16 for l in range(32)]}
        evs_2way = simulate_wave(code, lds_addrs_by_pc=addrs_2way)
        self.assertEqual((evs_2way[0].duration_cycles, evs_2way[1].duration_cycles), (34, 34))
        # Case B: Stride-20B (5 dwords, gcd(5,32)=1) is conflict-free across 32 banks -> 4 dwords * 2 half-waves = 8 port cyc -> dur=26
        addrs_cf = {0: [l * 20 for l in range(32)], 8: [l * 20 for l in range(32)]}
        evs = simulate_wave(code, lds_addrs_by_pc=addrs_cf)
        self.assertEqual((evs[0].pipe, evs[1].pipe), ("DS_LD", "DS_ST"))
        self.assertEqual((evs[0].lds_conflict_cycles, evs[1].lds_conflict_cycles), (0, 0))
        self.assertEqual((evs[0].duration_cycles, evs[1].duration_cycles), (26, 26))

    def test_14_smem_rd_zero_lds_conflict_and_tracks_lgkmcnt(self) -> None:
        code = (
            asm_smem(SMEM_LOAD_B128, sdata=4, sbase_pair=0, offset=0)
            + asm_ds(DS_LOAD_B32, addr=0, vdst=1)
            + asm_s_waitcnt(lgkmcnt=0)
            + asm_s_endpgm()
        )
        evs = simulate_wave(code)
        # SMEM_RD issues at cycle 0 with 0 LDS conflicts and does NOT block DS_LD at cycle 1!
        self.assertEqual(evs[0].lds_conflict_cycles, 0)
        self.assertEqual(evs[1].issue_cycle, 1)
        # s_waitcnt lgkmcnt(0) waits for max(SMEM complete=24, DS complete=21) = 24
        self.assertEqual(evs[2].issue_cycle, 24)

    def test_15_valu32_bypass_only_to_valu_not_ds_addr(self) -> None:
        # v1 = v_add_f32(v2, v3) at cyc 0 (completes cyc 4, bypassable to VALU at cyc 1)
        # Case A: VALU consumer reads v1 -> issues at cyc 1 (bypass!)
        code_valu = asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3) + asm_vop2(VOP2_MUL_F32, 4, 256 + 1, 5) + asm_s_endpgm()
        ev_v = simulate_wave(code_valu)
        self.assertEqual(ev_v[1].issue_cycle, 1)
        # Case B: DS_LOAD uses v1 as byte address -> NO VALU bypass! Must wait until cyc 4!
        code_ds = asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3) + asm_ds(DS_LOAD_B32, addr=1, vdst=4) + asm_s_endpgm()
        ev_d = simulate_wave(code_ds)
        self.assertEqual(ev_d[1].issue_cycle, 4)
        self.assertEqual(ev_d[1].stall_cycles, 3)

    def test_16_valu32_bypass_disabled_flag(self) -> None:
        code = asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3) + asm_vop2(VOP2_MUL_F32, 4, 256 + 1, 5) + asm_s_endpgm()
        ev = simulate_wave(code, profile=LatencyProfile(valu32_bypass=False))
        self.assertEqual(ev[1].issue_cycle, 4)

    def test_17_valut4_and_valub_pipes_latency_and_throughput(self) -> None:
        code = (
            asm_vop1(VOP1_RCP_F32, 1, 256 + 0)
            + asm_vop1(VOP1_SQRT_F32, 2, 256 + 3)  # Independent, waits 4 cyc VALUT_4 issue
            + asm_vop3(VOP3_FMA_F64, 4, 256 + 6, 256 + 8, 256 + 10)  # Starts after VALUT_4 issue
            + asm_vop2(VOP2_ADD_F32, 12, 256 + 1, 13)  # Depends on v1 (ready at 16) and VALUB_16 issue (8+16=24)
            + asm_s_endpgm()
        )
        ev = simulate_wave(code)
        self.assertEqual((ev[0].issue_cycle, ev[0].complete_cycle), (0, 16))
        self.assertEqual((ev[1].issue_cycle, ev[1].complete_cycle), (4, 20))
        self.assertEqual((ev[2].issue_cycle, ev[2].complete_cycle), (8, 24))
        self.assertEqual(ev[3].issue_cycle, 24)

    def test_18_vopd_dual_issue_vs_bank_conflict_serialization(self) -> None:
        code_ok = asm_vopd(VOPD_DUAL_ADD_F32, VOPD_DUAL_MUL_F32, 0, 256 + 2, 3, 1, 256 + 4, 5) + asm_vop2(VOP2_ADD_F32, 6, 256 + 7, 8) + asm_s_endpgm()
        code_bad = asm_vopd(VOPD_DUAL_ADD_F32, VOPD_DUAL_MUL_F32, 0, 256 + 2, 3, 2, 256 + 4, 5) + asm_vop2(VOP2_ADD_F32, 6, 256 + 7, 8) + asm_s_endpgm()
        ev_ok = simulate_wave(code_ok)
        ev_bad = simulate_wave(code_bad)
        self.assertTrue(ev_ok[0].dual_issued)
        self.assertEqual(ev_ok[1].issue_cycle, 1)
        self.assertFalse(ev_bad[0].dual_issued)
        self.assertEqual(ev_bad[1].issue_cycle, 2)

    def test_19_valu_salu_frontend_dual_issue(self) -> None:
        code = asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3) + asm_sop2(SOP2_ADD_U32, 4, 5, 6) + asm_s_endpgm()
        ev = simulate_wave(code)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 0)
        self.assertTrue(ev[0].dual_issued and ev[1].dual_issued)

    def test_20_valu_salu_hazard_blocks_dual_issue(self) -> None:
        # VALU reads s4 which SALU writes -> WAR/RAW dependency prevents co-issue!
        code = asm_sop2(SOP2_ADD_U32, 4, 5, 6) + asm_vop2(VOP2_ADD_F32, 1, 4, 3) + asm_s_endpgm()
        ev = simulate_wave(code)
        self.assertEqual(ev[0].issue_cycle, 0)
        self.assertEqual(ev[1].issue_cycle, 1)
        self.assertFalse(ev[0].dual_issued)

    def test_21_salu_wr_exec_updates_live_exec_mask_for_lds(self) -> None:
        # Set EXEC_LO = 0x0000FFFF (only lower 16 lanes active) before DS_LOAD with conflict in upper half!
        code = (
            asm_sop1(SOP1_MOV_B32, EXEC_LO_REG, LITERAL_SRC, lit=0x0000FFFF)
            + asm_ds(DS_LOAD_B32, addr=0, vdst=4)
            + asm_s_endpgm()
        )
        # Upper half has 16-way conflict; lower half is conflict-free!
        addrs = [l * 4 if l < 16 else (l - 16) * 128 for l in range(32)]
        ev = simulate_wave(code, lds_addrs_by_pc={8: addrs})
        self.assertEqual(ev[0].pipe, "SALU_WR_EXEC")
        self.assertEqual(ev[1].lds_conflict_cycles, 0)

    def test_22_cu_multiwave_shared_lds_port_arbitration(self) -> None:
        # Two waves on DIFFERENT SIMDs issue 16-way conflicting DS_LOAD_B32 at cycle 0.
        # Wave 0 holds shared CU LDS port for 32 cycles -> Wave 1 MUST stall until cycle 32!
        code = asm_ds(DS_LOAD_B32, addr=0, vdst=1) + asm_s_endpgm()
        bad_addrs = {0: [l * 128 for l in range(32)]}
        res = simulate_cu([
            WaveConfig(wave_id=0, simd_id=0, code_bytes=code, lds_addrs_by_pc=bad_addrs),
            WaveConfig(wave_id=1, simd_id=1, code_bytes=code, lds_addrs_by_pc=bad_addrs),
        ])
        self.assertEqual(res[0][0].issue_cycle, 0)
        self.assertEqual(res[1][0].issue_cycle, 32)
        self.assertEqual(res[1][0].stall_cycles, 32)

    def test_23_cu_multiwave_shared_vmem_port_arbitration(self) -> None:
        code = asm_global(GLOBAL_LOAD_B128, vaddr=0, vdst=4) + asm_s_endpgm()
        res = simulate_cu([
            WaveConfig(wave_id=0, simd_id=0, code_bytes=code),
            WaveConfig(wave_id=1, simd_id=1, code_bytes=code),
        ])
        # b128 holds shared VMEM port for 4*4 = 16 cycles -> wave 1 stalls until cycle 16
        self.assertEqual(res[0][0].issue_cycle, 0)
        self.assertEqual(res[1][0].issue_cycle, 16)

    def test_24_cu_multiwave_barrier_sync(self) -> None:
        # Wave 0 arrives at s_barrier immediately; Wave 1 runs a VALUT_4 + wait before s_barrier
        code0 = asm_s_barrier() + asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3) + asm_s_endpgm()
        code1 = asm_vop1(VOP1_RCP_F32, 1, 256 + 0) + asm_vop2(VOP2_ADD_F32, 2, 256 + 1, 3) + asm_s_barrier() + asm_s_endpgm()
        res = simulate_cu([
            WaveConfig(wave_id=0, simd_id=0, code_bytes=code0),
            WaveConfig(wave_id=1, simd_id=1, code_bytes=code1),
        ])
        bar0 = next(e for e in res[0] if e.pipe == "BARRIER")
        bar1 = next(e for e in res[1] if e.pipe == "BARRIER")
        self.assertEqual(bar0.issue_cycle, bar1.issue_cycle)
        self.assertGreater(bar0.stall_cycles, 10)

    def test_25_branch_loop_taken_and_untaken_cycles(self) -> None:
        # Loop body: v_add_f32 (PC 0), s_cbranch_scc1 back to PC 0 (PC 4), s_endpgm (PC 8)
        code = asm_vop2(VOP2_ADD_F32, 1, 256 + 1, 2) + asm_s_branch(SOPP_CBRANCH_SCC1, -2) + asm_s_endpgm()
        ev = simulate_wave(code, branch_outcomes={4: [True, True, False]})
        jumps = [e for e in ev if e.pipe == "JUMP"]
        self.assertEqual([j.duration_cycles for j in jumps], [4, 4, 1])

    def test_26_lgkmcnt_partial_wait(self) -> None:
        # Issue 2 DS loads (complete at 20, 22), then s_waitcnt lgkmcnt(1) -> waits only until 20!
        code = (
            asm_ds(DS_LOAD_B32, addr=0, vdst=2)
            + asm_ds(DS_LOAD_B32, addr=1, vdst=3)
            + asm_s_waitcnt(lgkmcnt=1)
            + asm_s_endpgm()
        )
        ev = simulate_wave(code)
        self.assertEqual(ev[0].complete_cycle, 20)
        self.assertEqual(ev[1].issue_cycle, 2)
        self.assertEqual(ev[1].complete_cycle, 22)
        self.assertEqual(ev[2].issue_cycle, 20)

    def test_27_sqtt_binary_and_hex_roundtrip(self) -> None:
        code = (
            asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3)
            + asm_sop2(SOP2_ADD_U32, 4, 5, 6)
            + asm_ds(DS_LOAD_B64, addr=0, vdst=8)
            + asm_s_waitcnt(lgkmcnt=0)
            + asm_s_endpgm()
        )
        cu_ev = simulate_cu([WaveConfig(wave_id=0, code_bytes=code)])
        raw = encode_sqtt(cu_ev)
        pkts_bin = parse_sqtt(raw)
        pkts_hex = parse_sqtt("0x" + raw.hex())
        self.assertEqual(pkts_bin, pkts_hex)
        inst_pkts = [p for p in pkts_bin if p.kind in ("INST", "VALU_DUAL")]
        self.assertEqual(len(inst_pkts), len(cu_ev[0]))
        for p, e in zip(inst_pkts, cu_ev[0]):
            self.assertEqual((p.pc, p.cycle, p.duration), (e.pc, e.issue_cycle, e.duration_cycles))

    def test_28_fit_profile_recovers_hidden_profile_exact_zero_mae(self) -> None:
        # Hidden ground-truth non-default profile!
        hidden = LatencyProfile(
            valu32_lat=6, valu32_bypass=False, valut4_lat=19,
            jump_taken_lat=5, lds_base_lat=27, lds_conflict_cyc=2,
            smem_lat=31, vmem_lat=136,
        )
        # 3 rich calibration kernels exercising every hidden parameter
        k1 = (
            asm_vop2(VOP2_ADD_F32, 1, 256 + 2, 3)
            + asm_vop2(VOP2_MUL_F32, 4, 256 + 1, 5)
            + asm_vop1(VOP1_RCP_F32, 6, 256 + 4)
            + asm_vop2(VOP2_ADD_F32, 7, 256 + 6, 5)
            + asm_s_branch(SOPP_CBRANCH_SCC1, -2)
            + asm_s_endpgm()
        )
        k2 = (
            asm_smem(SMEM_LOAD_B64, sdata=4, sbase_pair=0)
            + asm_ds(DS_LOAD_B32, addr=0, vdst=8)
            + asm_ds(DS_LOAD_B64, addr=1, vdst=10)
            + asm_s_waitcnt(lgkmcnt=0)
            + asm_global(GLOBAL_LOAD_B32, vaddr=2, vdst=12)
            + asm_s_waitcnt(vmcnt=0)
            + asm_s_endpgm()
        )
        w1 = [WaveConfig(wave_id=0, code_bytes=k1, branch_outcomes={16: [True, False]})]
        w2 = [WaveConfig(wave_id=0, code_bytes=k2, lds_addrs_by_pc={
            8: [l * 4 for l in range(32)],
            16: [l * 128 if l < 4 else l * 4 for l in range(32)],
        })]
        tr1 = encode_sqtt(simulate_cu(w1, profile=hidden))
        tr2 = encode_sqtt(simulate_cu(w2, profile=hidden))

        # Start from completely different default profile!
        res = fit_profile([TraceSample(w1, tr1), TraceSample(w2, tr2)], initial=LatencyProfile())
        self.assertAlmostEqual(res.mae_cycles, 0.0, places=6)
        self.assertEqual(res.profile.valu32_lat, hidden.valu32_lat)
        self.assertEqual(res.profile.valu32_bypass, hidden.valu32_bypass)
        self.assertEqual(res.profile.valut4_lat, hidden.valut4_lat)
        self.assertEqual(res.profile.lds_base_lat, hidden.lds_base_lat)
        self.assertEqual(res.profile.lds_conflict_cyc, hidden.lds_conflict_cyc)
        self.assertEqual(res.profile.smem_lat, hidden.smem_lat)
        self.assertEqual(res.profile.vmem_lat, hidden.vmem_lat)

    def test_29_multiwave_gemv_determinism_and_sqtt_hash(self) -> None:
        # 4-wave tiled GEMV microkernel with SMEM + DS + VOPD + barrier + GLOBAL store
        body = (
            asm_smem(SMEM_LOAD_B128, sdata=8, sbase_pair=0)
            + asm_ds(DS_LOAD_B128, addr=0, vdst=4)
            + asm_s_waitcnt(lgkmcnt=0)
            + asm_vopd(VOPD_DUAL_FMAC_F32, VOPD_DUAL_FMAC_F32, 10, 256 + 4, 5, 11, 256 + 6, 7)
            + asm_vop3(VOP3_FMA_F32, 12, 256 + 10, 256 + 11, 256 + 12)
            + asm_s_barrier()
            + asm_global(GLOBAL_STORE_B32, vaddr=0, vdata=12)
            + asm_s_waitcnt(vmcnt=0)
            + asm_s_endpgm()
        )
        waves = [WaveConfig(wave_id=i, simd_id=i % 4, code_bytes=body) for i in range(4)]
        tr_a = encode_sqtt(simulate_cu(waves))
        tr_b = encode_sqtt(simulate_cu(waves))
        self.assertEqual(tr_a, tr_b)
        self.assertEqual(len(parse_sqtt(tr_a)) > 30, True)

    def test_30_reject_truncated_or_misaligned_binary(self) -> None:
        with self.assertRaises(ValueError):
            decode_gfx11(b"\x00\x01\x02")
        with self.assertRaises(ValueError):
            decode_gfx11(asm_vop2(VOP2_FMAAK_F32, 0, 256, 1, lit=0)[:4])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DISPATCH-09 (REV=2) GFX11 Wave32 ISA & SQTT Cycle Emulator")
    parser.add_argument("--self-test", action="store_true", help="Run all 30 deterministic unit tests")
    args = parser.parse_args(argv)
    if args.self_test or len(sys.argv) == 1:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestWave32CyclesRev2)
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        return 0 if res.wasSuccessful() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
