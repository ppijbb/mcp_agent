#!/usr/bin/env python3
"""
Ethereum Trading MCP Server
Provides trading execution capabilities for Ethereum
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from web3 import Web3
from web3.middleware import geth_poa_middleware
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthereumTradingMCP:
    def __init__(self):
        self.w3 = None
        self.account = None
        self.setup_web3()
    
    def setup_web3(self):
        """Setup Web3 connection and account"""
        try:
            # Connect to Ethereum network (mainnet or testnet)
            rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID')
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add POA middleware for testnets
            if 'testnet' in rpc_url.lower():
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Setup account
            private_key = os.getenv('ETHEREUM_PRIVATE_KEY')
            if private_key:
                self.account = self.w3.eth.account.from_key(private_key)
                logger.info(f"Account loaded: {self.account.address}")
            
        except Exception as e:
            logger.error(f"Failed to setup Web3: {e}")
            raise
    
    async def get_balance(self, address: str) -> Dict[str, Any]:
        """Get ETH balance for address"""
        try:
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return {
                "address": address,
                "balance_wei": str(balance_wei),
                "balance_eth": str(balance_eth),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_gas_price(self) -> Dict[str, Any]:
        """Get current gas price"""
        try:
            gas_price = self.w3.eth.gas_price
            gas_price_gwei = self.w3.from_wei(gas_price, 'gwei')
            return {
                "gas_price_wei": str(gas_price),
                "gas_price_gwei": str(gas_price_gwei),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return {"status": "error", "message": str(e)}
    
    async def send_transaction(self, to_address: str, amount_eth: float, gas_limit: int = 21000) -> Dict[str, Any]:
        """Send ETH transaction"""
        try:
            if not self.account:
                return {"status": "error", "message": "No account configured"}
            
            # Convert amount to Wei
            amount_wei = self.w3.to_wei(amount_eth, 'ether')
            
            # Get gas price
            gas_price = self.w3.eth.gas_price
            
            # Build transaction
            transaction = {
                'to': to_address,
                'value': amount_wei,
                'gas': gas_limit,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'chainId': self.w3.eth.chain_id
            }
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return {
                "status": "success",
                "tx_hash": tx_hash.hex(),
                "from": self.account.address,
                "to": to_address,
                "amount_eth": str(amount_eth),
                "gas_used": str(gas_limit),
                "gas_price_gwei": str(self.w3.from_wei(gas_price, 'gwei'))
            }
            
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status and details"""
        try:
            tx_receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            if tx_receipt:
                return {
                    "status": "success",
                    "tx_hash": tx_hash,
                    "block_number": tx_receipt.blockNumber,
                    "gas_used": str(tx_receipt.gasUsed),
                    "status_code": tx_receipt.status,
                    "confirmed": tx_receipt.status == 1
                }
            else:
                return {"status": "pending", "tx_hash": tx_hash}
                
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {"status": "error", "message": str(e)}

# MCP Server implementation
async def main():
    trading_mcp = EthereumTradingMCP()
    
    # Example usage
    if trading_mcp.account:
        balance = await trading_mcp.get_balance(trading_mcp.account.address)
        print(f"Balance: {balance}")
        
        gas_price = await trading_mcp.get_gas_price()
        print(f"Gas Price: {gas_price}")

if __name__ == "__main__":
    asyncio.run(main())
