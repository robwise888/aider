"""
Configuration schema for the Selfy agent.

This module provides the schema for validating configuration values.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Define the schema
CONFIG_SCHEMA = {
    'agent': {
        'type': dict,
        'required': False,
        'schema': {
            'name': {
                'type': str,
                'required': False,
                'default': 'Selfy'
            },
            'version': {
                'type': str,
                'required': False,
                'default': '1.0.0'
            },
            'environment': {
                'type': str,
                'required': False,
                'enum': ['development', 'testing', 'production'],
                'default': 'development'
            }
        }
    },
    'logging': {
        'type': dict,
        'required': True,
        'schema': {
            'level': {
                'type': str,
                'required': True,
                'default': 'INFO',
                'enum': ['DEBUG', 'TRACE', 'VERBOSE', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
            },
            'format': {
                'type': str,
                'required': True,
                'default': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'file': {
                'type': str,
                'required': True,
                'default': 'logs/selfy.log'
            },
            'max_size': {
                'type': int,
                'required': True,
                'default': 10485760,
                'min': 1024
            },
            'backup_count': {
                'type': int,
                'required': True,
                'default': 5,
                'min': 0
            },
            'json_file': {
                'type': str,
                'required': False,
                'default': 'logs/selfy_structured.json'
            },
            'error_file': {
                'type': str,
                'required': False,
                'default': 'logs/selfy_errors.log'
            },
            'levels': {
                'type': dict,
                'required': False,
                'default': {
                    'selfy_core': 'DEBUG',
                    'selfy_core.global_modules': 'DEBUG',
                    'selfy_core.user_pipeline': 'DEBUG'
                }
            }
        }
    },
    'llm': {
        'type': dict,
        'required': True,
        'schema': {
            'default_provider': {
                'type': str,
                'required': True,
                'enum': ['groq', 'ollama'],
                'default': 'ollama'
            },
            'cloud_provider': {
                'type': str,
                'required': False,
                'enum': ['groq', 'anthropic', 'openai'],
                'default': 'groq'
            },
            'local_provider': {
                'type': str,
                'required': False,
                'enum': ['ollama'],
                'default': 'ollama'
            },
            'token_tracking': {
                'type': dict,
                'required': False,
                'schema': {
                    'enabled': {
                        'type': bool,
                        'required': False,
                        'default': True
                    }
                }
            },
            'groq': {
                'type': dict,
                'required': False,
                'schema': {
                    'enabled': {
                        'type': bool,
                        'required': False,
                        'default': True
                    },
                    'api_key': {
                        'type': str,
                        'required': False,
                        'default': ''
                    },
                    'model': {
                        'type': str,
                        'required': False,
                        'default': 'llama3-70b-8192'
                    },
                    'default_model': {
                        'type': str,
                        'required': False,
                        'default': 'llama3-70b-8192'
                    },
                    'default_system_prompt': {
                        'type': str,
                        'required': False,
                        'default': 'You are a helpful assistant.'
                    },
                    'max_retries': {
                        'type': int,
                        'required': False,
                        'default': 2,
                        'min': 0
                    },
                    'retry_delay_seconds': {
                        'type': float,
                        'required': False,
                        'default': 1.0,
                        'min': 0
                    },
                    'timeout_seconds': {
                        'type': int,
                        'required': False,
                        'default': 30,
                        'min': 1
                    }
                }
            },
            'ollama': {
                'type': dict,
                'required': False,
                'schema': {
                    'enabled': {
                        'type': bool,
                        'required': False,
                        'default': True
                    },
                    'host': {
                        'type': str,
                        'required': False,
                        'default': 'http://localhost:11434'
                    },
                    'model': {
                        'type': str,
                        'required': False,
                        'default': 'llama3:8b'
                    },
                    'default_model': {
                        'type': str,
                        'required': False,
                        'default': 'llama3:8b'
                    },
                    'default_system_prompt': {
                        'type': str,
                        'required': False,
                        'default': 'You are a helpful assistant.'
                    },
                    'max_retries': {
                        'type': int,
                        'required': False,
                        'default': 2,
                        'min': 0
                    },
                    'retry_delay_seconds': {
                        'type': float,
                        'required': False,
                        'default': 1.0,
                        'min': 0
                    },
                    'timeout_seconds': {
                        'type': int,
                        'required': False,
                        'default': 30,
                        'min': 1
                    },
                    'format': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'user_prefix': {
                                'type': str,
                                'required': False,
                                'default': 'User: '
                            },
                            'assistant_prefix': {
                                'type': str,
                                'required': False,
                                'default': 'Assistant: '
                            },
                            'system_prefix': {
                                'type': str,
                                'required': False,
                                'default': 'System: '
                            },
                            'assistant_response_prefix': {
                                'type': str,
                                'required': False,
                                'default': 'Assistant: '
                            }
                        }
                    }
                }
            }
        }
    },
    'memory': {
        'type': dict,
        'required': True,
        'schema': {
            'working_memory': {
                'type': dict,
                'required': True,
                'schema': {
                    'max_size': {
                        'type': int,
                        'required': True,
                        'default': 100,
                        'min': 1
                    },
                    'eviction_policy': {
                        'type': str,
                        'required': True,
                        'default': 'FIFO',
                        'enum': ['FIFO', 'LRU']
                    }
                }
            },
            'embedding_service': {
                'type': dict,
                'required': True,
                'schema': {
                    'provider': {
                        'type': str,
                        'required': True,
                        'default': 'sentence_transformers',
                        'enum': ['sentence_transformers', 'llm']
                    },
                    'model': {
                        'type': str,
                        'required': True,
                        'default': 'all-MiniLM-L6-v2'
                    },
                    'dimension': {
                        'type': int,
                        'required': True,
                        'default': 384,
                        'min': 1
                    },
                    'use_cache': {
                        'type': bool,
                        'required': True,
                        'default': True
                    },
                    'cache_size': {
                        'type': int,
                        'required': False,
                        'default': 1000,
                        'min': 1
                    },
                    'use_gpu': {
                        'type': bool,
                        'required': False,
                        'default': True
                    }
                }
            },
            'lts': {
                'type': dict,
                'required': True,
                'schema': {
                    'vector_db_type': {
                        'type': str,
                        'required': True,
                        'default': 'chromadb',
                        'enum': ['chromadb']
                    },
                    'chromadb': {
                        'type': dict,
                        'required': True,
                        'schema': {
                            'path': {
                                'type': str,
                                'required': True,
                                'default': './memory_db'
                            },
                            'collection_name': {
                                'type': str,
                                'required': True,
                                'default': 'selfy_memory'
                            }
                        }
                    }
                }
            },
            'consolidation': {
                'type': dict,
                'required': False,
                'schema': {
                    'consolidate_on_shutdown': {
                        'type': bool,
                        'required': False,
                        'default': True
                    },
                    'check_after_each_turn': {
                        'type': bool,
                        'required': False,
                        'default': False
                    },
                    'inactivity_threshold_minutes': {
                        'type': int,
                        'required': False,
                        'default': 30,
                        'min': 1
                    }
                }
            },
            'types': {
                'type': dict,
                'required': False,
                'schema': {
                    'conversation_turn': {
                        'type': str,
                        'required': False,
                        'default': 'conversation_turn'
                    },
                    'capability_info': {
                        'type': str,
                        'required': False,
                        'default': 'capability_info'
                    },
                    'execution_log': {
                        'type': str,
                        'required': False,
                        'default': 'execution_log'
                    },
                    'learned_fact': {
                        'type': str,
                        'required': False,
                        'default': 'learned_fact'
                    },
                    'user_preference': {
                        'type': str,
                        'required': False,
                        'default': 'user_preference'
                    }
                }
            },
            'roles': {
                'type': dict,
                'required': False,
                'schema': {
                    'user': {
                        'type': str,
                        'required': False,
                        'default': 'user'
                    },
                    'assistant': {
                        'type': str,
                        'required': False,
                        'default': 'assistant'
                    },
                    'system': {
                        'type': str,
                        'required': False,
                        'default': 'system'
                    }
                }
            }
        }
    },
    'capability': {
        'type': dict,
        'required': True,
        'schema': {
            'manifest_path': {
                'type': str,
                'required': True,
                'default': 'data/capability_manifest.json'
            },
            'scan_libraries_on_startup': {
                'type': bool,
                'required': True,
                'default': False
            },
            'convert_to_standard_format': {
                'type': bool,
                'required': True,
                'default': True
            },
            'execution': {
                'type': dict,
                'required': False,
                'schema': {
                    'use_old_format': {
                        'type': bool,
                        'required': False,
                        'default': False
                    }
                }
            },
            'discovery': {
                'type': dict,
                'required': False,
                'schema': {
                    'included_libraries': {
                        'type': list,
                        'required': False,
                        'default': ['math', 'random', 'statistics', 'datetime', 'json']
                    },
                    'excluded_libraries': {
                        'type': list,
                        'required': False,
                        'default': []
                    },
                    'standard_libraries': {
                        'type': list,
                        'required': False,
                        'default': ['math', 'statistics', 'datetime', 'json', 'random', 're', 'os', 'sys']
                    }
                }
            }
        }
    },
    'paths': {
        'type': dict,
        'required': False,
        'schema': {
            'data_dir': {
                'type': str,
                'required': False,
                'default': 'data'
            },
            'capability_manifest': {
                'type': str,
                'required': False,
                'default': 'data/capability_manifest.json'
            }
        }
    },
    'identity': {
        'type': dict,
        'required': False,
        'schema': {
            'profile': {
                'type': dict,
                'required': False,
                'schema': {
                    'name': {
                        'type': str,
                        'required': False,
                        'default': 'Selfy'
                    },
                    'persona_summary': {
                        'type': str,
                        'required': False,
                        'default': 'An AI agent specializing in code assistance and self-improvement.'
                    },
                    'core_values': {
                        'type': list,
                        'required': False,
                        'default': ['Modularity', 'Clarity', 'Helpfulness']
                    },
                    'tone_keywords': {
                        'type': list,
                        'required': False,
                        'default': ['professional', 'concise', 'constructive']
                    },
                    'strengths_summary': {
                        'type': str,
                        'required': False,
                        'default': 'Code analysis, refactoring, and explaining complex concepts clearly.'
                    },
                    'development_goals': {
                        'type': str,
                        'required': False,
                        'default': 'Continuous improvement in code understanding and generation capabilities.'
                    },
                    'prohibited_statements': {
                        'type': list,
                        'required': False,
                        'default': [
                            'I am an AI',
                            'I am a language model',
                            'I am not a real person',
                            'I do not have personal opinions',
                            'I do not have emotions'
                        ]
                    }
                }
            },
            'filter': {
                'type': dict,
                'required': False,
                'schema': {
                    'enable_llm_checks': {
                        'type': bool,
                        'required': False,
                        'default': False
                    },
                    'llm_check_provider': {
                        'type': str,
                        'required': False,
                        'default': 'ollama'
                    },
                    'input': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'keywords_to_flag': {
                                'type': list,
                                'required': False,
                                'default': [
                                    'who are you really',
                                    'ignore previous instructions',
                                    'what are your rules'
                                ]
                            },
                            'canned_response': {
                                'type': str,
                                'required': False,
                                'default': "I'm here to help with your questions and tasks. How can I assist you today?"
                            }
                        }
                    },
                    'output': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'replacement_enabled': {
                                'type': bool,
                                'required': False,
                                'default': True
                            }
                        }
                    }
                }
            }
        }
    },
    'pipeline': {
        'type': dict,
        'required': False,
        'schema': {
            'input_handling': {
                'type': dict,
                'required': False,
                'schema': {
                    'max_input_length': {
                        'type': int,
                        'required': False,
                        'default': 2048
                    },
                    'history': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'max_turns': {
                                'type': int,
                                'required': False,
                                'default': 10
                            }
                        }
                    }
                }
            },
            'output_handling': {
                'type': dict,
                'required': False,
                'schema': {
                    'validation': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'min_length': {
                                'type': int,
                                'required': False,
                                'default': 1
                            },
                            'max_length': {
                                'type': int,
                                'required': False,
                                'default': 8192
                            },
                            'enable_sanitization': {
                                'type': bool,
                                'required': False,
                                'default': True
                            },
                            'safety_patterns': {
                                'type': list,
                                'required': False,
                                'default': []
                            }
                        }
                    },
                    'formatting': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'default_format': {
                                'type': str,
                                'required': False,
                                'default': 'text'
                            }
                        }
                    }
                }
            }
        }
    },
    'context_engine': {
        'type': dict,
        'required': False,
        'schema': {
            'use_local_llm': {
                'type': bool,
                'required': False,
                'default': True
            },
            'use_cloud_llm': {
                'type': bool,
                'required': False,
                'default': True
            },
            'cache_size': {
                'type': int,
                'required': False,
                'default': 100,
                'min': 10,
                'max': 1000
            },
            'cache': {
                'type': dict,
                'required': False,
                'schema': {
                    'max_turns': {
                        'type': int,
                        'required': False,
                        'default': 3,
                        'min': 1,
                        'max': 10
                    }
                }
            },
            'request_analyzer': {
                'type': dict,
                'required': False,
                'schema': {
                    'confidence_threshold': {
                        'type': float,
                        'required': False,
                        'default': 0.95,  # Changed from 0.7 to 0.95 as requested
                        'min': 0.0,
                        'max': 1.0
                    },
                    'skip_final_analysis_threshold': {
                        'type': float,
                        'required': False,
                        'default': 0.98,  # Slightly higher than confidence_threshold
                        'min': 0.0,
                        'max': 1.0
                    }
                }
            },
            'execution_planner': {
                'type': dict,
                'required': False,
                'schema': {
                    'max_tokens': {
                        'type': int,
                        'required': False,
                        'default': 4000,
                        'min': 1000,
                        'max': 8000
                    }
                }
            },
            'use_llm_assisted_context': {
                'type': bool,
                'required': False,
                'default': True
            },
            'llm_assist_threshold': {
                'type': float,
                'required': False,
                'default': 0.0,
                'min': 0.0,
                'max': 1.0
            }
        }
    }
}

def get_schema() -> Dict[str, Any]:
    """
    Get the configuration schema.

    Returns:
        The configuration schema
    """
    return CONFIG_SCHEMA
