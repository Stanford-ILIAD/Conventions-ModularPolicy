from partner import Partner, PartnerPolicy, PPOPartnerPolicy, ArmsPartnerPolicy, BlocksPermutationPartnerPolicy

def get_arms_partners(setting, partner_type):
    partners_dict = {
        "n4m0": {
            "fixed": {
                "train": [
                    [0,1,2,3],
                    [4,1,6,7],
                    [0,5,2,7],
                    [4,5,6,3],
                ],
                "test": [
                    [0,1,2,7],
                    [4,1,6,3],
                    [0,5,2,3],
                    [4,5,6,7],
                ],
                "inverttrain": [
                    [0,1,2,7],
                    [4,1,6,3],
                    [0,5,2,3],
                    [4,5,6,7],
                ],
                "inverttest": [
                    [0,1,2,7],
                    [4,1,6,3],
                    [0,5,2,3],
                    [4,5,6,7],
                ],
            },
            "ppo": {
                "train": [
                    "output/arms_n=4_m=0_run=1240_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1241_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1242_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1243_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1244_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1245_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1246_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1247_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1248_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1249_netsz=030_mreg=0.00",
                ],
                "test": [
                    "output/arms_n=4_m=0_run=1230_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1231_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1232_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1233_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1234_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1235_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1236_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1237_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1238_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=0_run=1239_netsz=030_mreg=0.00",
                ]
            }
        },
        "n4m1": {
            "fixed": {
                "train": [
                    [0,1,2,3],
                    [0,1,6,7],
                    [0,5,2,7],
                    [0,5,6,3],
                ],
                "test": [
                    [0,1,2,7],
                    [0,1,6,3],
                    [0,5,2,3],
                    [0,5,6,7],
                ],
                "inverttrain": [
                    [4,1,2,3],
                    [4,1,6,7],
                    [4,5,2,7],
                    [4,5,6,3],
                ],
                "inverttest": [
                    [4,1,2,7],
                    [4,1,6,3],
                    [4,5,2,3],
                    [4,5,6,7],
                ],
            },
            "ppo": {
                "train": [
                    "output/arms_n=4_m=1_run=1240_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1241_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1242_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1243_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1244_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1245_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1246_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1247_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1248_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1249_netsz=030_mreg=0.00",
                ],
                "test": [
                    "output/arms_n=4_m=1_run=1230_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1231_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1232_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1233_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1234_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1235_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1236_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1237_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1238_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=1_run=1239_netsz=030_mreg=0.00",
                ]
            }
        },
        "n4m2": {
            "fixed": {
                "train": [
                    [0,1,2,3],
                    [0,1,6,7],
                    [0,1,2,7],
                    [0,1,6,3],
                ],
                "test": [
                    [0,1,2,7],
                    [0,1,6,3],
                    [0,1,2,3],
                    [0,1,6,7],
                ],
                "inverttrain": [
                    [4,5,2,3],
                    [4,5,6,7],
                    [4,5,2,7],
                    [4,5,6,3],
                ],
                "inverttest": [
                    [4,5,2,7],
                    [4,5,6,3],
                    [4,5,2,3],
                    [4,5,6,7],
                ],
            },
            "ppo": {
                "train": [
                    "output/arms_n=4_m=2_run=1240_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1241_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1242_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1243_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1244_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1245_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1246_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1247_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1248_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1249_netsz=030_mreg=0.00",
                ],
                "test": [
                    "output/arms_n=4_m=2_run=1230_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1231_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1232_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1233_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1234_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1235_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1236_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1237_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1238_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=2_run=1239_netsz=030_mreg=0.00",
                ]
            }
        },
        "n4m3": {
            "fixed": {
                "train": [
                    [0,1,2,3],
                    [0,1,2,7],
                    [0,1,2,7],
                    [0,1,2,3],
                ],
                "test": [
                    [0,1,2,7],
                    [0,1,2,3],
                    [0,1,2,3],
                    [0,1,2,7],
                ],
                "inverttrain": [
                    [4,5,6,3],
                    [4,5,6,7],
                    [4,5,6,7],
                    [4,5,6,3],
                ],
                "inverttest": [
                    [4,5,6,7],
                    [4,5,6,3],
                    [4,5,6,3],
                    [4,5,6,7],
                ],
            },
            "ppo": {
                "train": [
                    "output/arms_n=4_m=3_run=1240_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1241_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1242_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1243_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1244_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1245_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1246_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1247_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1248_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1249_netsz=030_mreg=0.00",
                ],
                "test": [
                    "output/arms_n=4_m=3_run=1230_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1231_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1232_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1233_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1234_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1235_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1236_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1237_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1238_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=3_run=1239_netsz=030_mreg=0.00",
                ]
            }
        },
        "n4m4": {
            "fixed": {
                "train": [
                    [0,1,2,3],
                    [0,1,2,3],
                    [0,1,2,3],
                    [0,1,2,3],
                ],
                "test": [
                    [0,1,2,3],
                    [0,1,2,3],
                    [0,1,2,3],
                    [0,1,2,3],
                ],
                "inverttrain": [
                    [4,5,6,7],
                    [4,5,6,7],
                    [4,5,6,7],
                    [4,5,6,7],
                ],
                "inverttest": [
                    [4,5,6,7],
                    [4,5,6,7],
                    [4,5,6,7],
                    [4,5,6,7],
                ],
            },
            "ppo": {
                "train": [
                    "output/arms_n=4_m=4_run=1240_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1241_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1242_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1243_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1244_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1245_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1246_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1247_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1248_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1249_netsz=030_mreg=0.00",
                ],
                "test": [
                    "output/arms_n=4_m=4_run=1230_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1231_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1232_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1233_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1234_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1235_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1236_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1237_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1238_netsz=030_mreg=0.00",
                    "output/arms_n=4_m=4_run=1239_netsz=030_mreg=0.00",
                ]
            }
        },
    }

    partners = partners_dict[setting][partner_type]
    if partner_type == "fixed":
        train_partners = [Partner(ArmsPartnerPolicy(perm=perm)) for perm in partners["train"]]
        test_partners = [Partner(ArmsPartnerPolicy(perm=perm)) for perm in partners["test"]]
        inverttrain_partners = [Partner(ArmsPartnerPolicy(perm=perm)) for perm in partners["inverttrain"]]
        inverttest_partners = [Partner(ArmsPartnerPolicy(perm=perm)) for perm in partners["inverttest"]]
    if partner_type == "ppo":
        train_partners = [Partner(PPOPartnerPolicy(pmpath)) for pmpath in partners["train"]]
        test_partners = [Partner(PPOPartnerPolicy(pmpath)) for pmpath in partners["test"]] 
        inverttrain_partners = None
        inverttest_partners = None
    
    return train_partners, test_partners, inverttrain_partners, inverttest_partners

def get_blocks_partners(setting, partner_type):
    partners_dict = {
        "": {
            "fixed": {
                "train": [
                    [1,3,0,2], # clockwise
                    [2,0,3,1], # counter-clockwise
                    [3,2,1,0], # diagonal
                ],
                "test": [
                    [1,0,3,2],
                    [2,3,1,0],
                    [3,2,0,1],
                    [1,2,3,0],
                    [2,3,0,1],
                    [3,0,1,2],
                ],
                "inverttrain": [
                    [1,3,0,2], # clockwise
                    [2,0,3,1], # counter-clockwise
                    [3,2,1,0], # diagonal
                ],
                "inverttest": [
                    [1,0,3,2],
                    [2,3,1,0],
                    [3,2,0,1],
                ],
            },
            "ppo": {
                "train": [
                    "output/blocks_n=2_run=1230_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1231_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1232_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1233_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1234_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1235_vis1=1_vis2=3_onesided=0_mreg=0.00",
                ],
                "test": [
                    "output/blocks_n=2_run=1240_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1241_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1242_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1243_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1244_vis1=1_vis2=3_onesided=0_mreg=0.00",
                    "output/blocks_n=2_run=1245_vis1=1_vis2=3_onesided=0_mreg=0.00",
                ]
            }
        }
    }
    partners = partners_dict[setting][partner_type]
    if partner_type == "fixed":
        train_partners = [Partner(BlocksPermutationPartnerPolicy(perm=perm)) for perm in partners["train"]]
        test_partners = [Partner(BlocksPermutationPartnerPolicy(perm=perm)) for perm in partners["test"]]
        inverttrain_partners = [Partner(BlocksPermutationPartnerPolicy(perm=perm)) for perm in partners["inverttrain"]]
        inverttest_partners = [Partner(BlocksPermutationPartnerPolicy(perm=perm)) for perm in partners["inverttest"]]
    if partner_type == "ppo":
        train_partners = [Partner(PPOPartnerPolicy(pmpath)) for pmpath in partners["train"]]
        test_partners = [Partner(PPOPartnerPolicy(pmpath)) for pmpath in partners["test"]] 
        inverttrain_partners = None
        inverttest_partners = None
    
    return train_partners, test_partners, inverttrain_partners, inverttest_partners


def get_hanabi_partners(setting, partner_type):
    partners_dict = {
        "": {
            "ppo": {
                "train": [
                    "output/hanabi_n=4_run=1240_netsz=500_mreg=0.00",
                    "output/hanabi_n=4_run=1241_netsz=500_mreg=0.00",
                    "output/hanabi_n=4_run=1242_netsz=500_mreg=0.00",
                    "output/hanabi_n=4_run=1243_netsz=500_mreg=0.00",
                ],
                "test": [
                    "output/hanabi_n=4_run=1244_netsz=500_mreg=0.00",
                    "output/hanabi_n=4_run=1245_netsz=500_mreg=0.00",
                    "output/hanabi_n=4_run=1246_netsz=500_mreg=0.00",
                    "output/hanabi_n=4_run=1247_netsz=500_mreg=0.00",
                ]
            }
        }
    }
    partners = partners_dict[setting][partner_type]
    if partner_type == "ppo":
        train_partners = [Partner(PPOPartnerPolicy(pmpath)) for pmpath in partners["train"]]
        test_partners = [Partner(PPOPartnerPolicy(pmpath)) for pmpath in partners["test"]] 
    
    return train_partners, test_partners


def get_arms_human_partners(setting, partner_type):
    partners_dict = {
        "": {
            "fixed": {
                "train": [
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 3, 3],
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 3, 3],
                    [0, 2, 1],
                    [0, 2, 1],
                ],
                "test": [
                    [0, 2, 1],
                    [0, 3, 3],
                    [0, 2, 1],
                    [0, 3, 3],
                    [0, 3, 1],
                    [0, 2, 1],
                    [0, 3, 3],
                    [0, 2, 1],
                    [0, 2, 1],
                    [0, 3, 1],
                ],
            },
        },
    }

    partners = partners_dict[setting][partner_type]
    if partner_type == "fixed":
        train_partners = [Partner(ArmsPartnerPolicy(perm=perm)) for perm in partners["train"]]
        test_partners = [Partner(ArmsPartnerPolicy(perm=perm)) for perm in partners["test"]]
    if partner_type == "ppo":
        exit(1)
    
    return train_partners, test_partners