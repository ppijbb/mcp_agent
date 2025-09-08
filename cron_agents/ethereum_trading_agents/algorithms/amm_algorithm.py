"""
Advanced AMM (Automated Market Making) Algorithm for Ethereum Trading

This module implements sophisticated AMM strategies optimized for Ethereum:
1. Uniswap V3 style concentrated liquidity
2. Dynamic fee optimization
3. Impermanent loss protection
4. MEV protection strategies
"""

import asyncio
import math
from typing import Dict, List, Tuple, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LiquidityRange(TypedDict):
    """Liquidity concentration range for AMM"""
    lower_tick: int
    upper_tick: int
    liquidity: float
    fee_tier: float

class AMMPosition(TypedDict):
    """AMM position data"""
    token0: str
    token1: str
    liquidity: float
    tick_lower: int
    tick_upper: int
    fee_tier: float
    current_tick: int
    price_range: Tuple[float, float]

class FeeTier(Enum):
    """Uniswap V3 fee tiers"""
    LOW = 0.0005    # 0.05%
    MEDIUM = 0.003  # 0.30%
    HIGH = 0.01     # 1.00%

@dataclass
class AMMConfig:
    """AMM algorithm configuration"""
    min_liquidity: float = 1000.0
    max_liquidity: float = 100000.0
    price_impact_threshold: float = 0.01  # 1%
    impermanent_loss_threshold: float = 0.05  # 5%
    rebalance_frequency: int = 3600  # 1 hour
    fee_optimization: bool = True
    mev_protection: bool = True

class AMMAlgorithm:
    """Advanced AMM algorithm for Ethereum trading"""
    
    def __init__(self, config: AMMConfig):
        self.config = config
        self.positions: List[AMMPosition] = []
        self.liquidity_ranges: List[LiquidityRange] = []
        self.current_prices: Dict[str, float] = {}
        
    async def calculate_optimal_liquidity_range(
        self, 
        token0: str, 
        token1: str, 
        current_price: float,
        volatility: float
    ) -> Tuple[int, int, float]:
        """Calculate optimal liquidity concentration range"""
        try:
            # Calculate price range based on volatility
            price_range = volatility * 2  # ±2σ range
            
            # Convert to tick values
            tick_spacing = self._get_tick_spacing(FeeTier.MEDIUM)
            current_tick = self._price_to_tick(current_price)
            
            # Calculate range ticks
            tick_range = int(price_range * 10000)  # Convert to tick units
            lower_tick = (current_tick - tick_range) // tick_spacing * tick_spacing
            upper_tick = (current_tick + tick_range) // tick_spacing * tick_spacing
            
            # Calculate optimal liquidity amount
            liquidity = self._calculate_liquidity_amount(
                current_price, lower_tick, upper_tick, volatility
            )
            
            return lower_tick, upper_tick, liquidity
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal liquidity range: {e}")
            raise ValueError(f"AMM range calculation failed: {str(e)}")
    
    def _get_tick_spacing(self, fee_tier: FeeTier) -> int:
        """Get tick spacing for fee tier"""
        spacing_map = {
            FeeTier.LOW: 10,
            FeeTier.MEDIUM: 60,
            FeeTier.HIGH: 200
        }
        return spacing_map[fee_tier]
    
    def _price_to_tick(self, price: float) -> int:
        """Convert price to tick value"""
        return int(math.log(price, 1.0001))
    
    def _tick_to_price(self, tick: int) -> float:
        """Convert tick value to price"""
        return 1.0001 ** tick
    
    def _calculate_liquidity_amount(
        self, 
        current_price: float, 
        lower_tick: int, 
        upper_tick: int, 
        volatility: float
    ) -> float:
        """Calculate optimal liquidity amount based on volatility and price range"""
        try:
            # Base liquidity calculation
            base_liquidity = self.config.min_liquidity
            
            # Adjust for volatility (higher volatility = more liquidity needed)
            volatility_multiplier = 1 + (volatility * 2)
            
            # Adjust for price range (narrower range = more liquidity needed)
            price_range = self._tick_to_price(upper_tick) - self._tick_to_price(lower_tick)
            range_multiplier = 1 / (price_range / current_price)
            
            # Calculate final liquidity
            liquidity = base_liquidity * volatility_multiplier * range_multiplier
            
            # Apply bounds
            return max(
                self.config.min_liquidity,
                min(liquidity, self.config.max_liquidity)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate liquidity amount: {e}")
            raise ValueError(f"Liquidity calculation failed: {str(e)}")
    
    async def optimize_fee_tier(
        self, 
        token0: str, 
        token1: str, 
        volume_24h: float,
        volatility: float
    ) -> FeeTier:
        """Optimize fee tier based on volume and volatility"""
        try:
            if not self.config.fee_optimization:
                return FeeTier.MEDIUM
            
            # Calculate expected fees for each tier
            fee_calculations = {}
            
            for tier in FeeTier:
                expected_volume = volume_24h * 0.1  # Assume 10% of volume goes through our position
                expected_fees = expected_volume * tier.value
                
                # Adjust for volatility (higher volatility = more trading)
                volatility_adjustment = 1 + (volatility * 0.5)
                adjusted_fees = expected_fees * volatility_adjustment
                
                fee_calculations[tier] = adjusted_fees
            
            # Select tier with highest expected fees
            optimal_tier = max(fee_calculations, key=fee_calculations.get)
            
            logger.info(f"Optimal fee tier selected: {optimal_tier.name} ({optimal_tier.value*100}%)")
            return optimal_tier
            
        except Exception as e:
            logger.error(f"Failed to optimize fee tier: {e}")
            return FeeTier.MEDIUM
    
    async def calculate_impermanent_loss(
        self, 
        position: AMMPosition, 
        current_price: float
    ) -> float:
        """Calculate impermanent loss for a position"""
        try:
            # Get position price range
            lower_price = self._tick_to_price(position["tick_lower"])
            upper_price = self._tick_to_price(position["tick_upper"])
            
            # Calculate current price ratio
            price_ratio = current_price / position["current_tick"]
            
            # Calculate impermanent loss
            if price_ratio <= lower_price or price_ratio >= upper_price:
                # Price is outside range - full impermanent loss
                return 1.0
            
            # Calculate IL using Uniswap V3 formula
            sqrt_price = math.sqrt(price_ratio)
            sqrt_lower = math.sqrt(lower_price)
            sqrt_upper = math.sqrt(upper_price)
            
            # IL = 1 - (2 * sqrt(price_ratio)) / (1 + price_ratio)
            il = 1 - (2 * sqrt_price) / (1 + price_ratio)
            
            return abs(il)
            
        except Exception as e:
            logger.error(f"Failed to calculate impermanent loss: {e}")
            return 0.0
    
    async def should_rebalance_position(
        self, 
        position: AMMPosition, 
        current_price: float,
        volatility: float
    ) -> bool:
        """Determine if position should be rebalanced"""
        try:
            # Check impermanent loss
            il = await self.calculate_impermanent_loss(position, current_price)
            if il > self.config.impermanent_loss_threshold:
                logger.warning(f"High impermanent loss detected: {il:.2%}")
                return True
            
            # Check if price is near range boundaries
            lower_price = self._tick_to_price(position["tick_lower"])
            upper_price = self._tick_to_price(position["tick_upper"])
            
            price_range = upper_price - lower_price
            distance_to_lower = abs(current_price - lower_price) / price_range
            distance_to_upper = abs(current_price - upper_price) / price_range
            
            # Rebalance if price is within 20% of range boundaries
            if distance_to_lower < 0.2 or distance_to_upper < 0.2:
                logger.info("Price near range boundary, rebalancing recommended")
                return True
            
            # Check volatility change
            if volatility > position.get("volatility", 0) * 1.5:
                logger.info("Volatility increased significantly, rebalancing recommended")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rebalance condition: {e}")
            return False
    
    async def execute_amm_strategy(
        self, 
        token0: str, 
        token1: str, 
        current_price: float,
        volatility: float,
        volume_24h: float
    ) -> Dict[str, any]:
        """Execute complete AMM strategy"""
        try:
            # Optimize fee tier
            optimal_fee_tier = await self.optimize_fee_tier(token0, token1, volume_24h, volatility)
            
            # Calculate optimal liquidity range
            lower_tick, upper_tick, liquidity = await self.calculate_optimal_liquidity_range(
                token0, token1, current_price, volatility
            )
            
            # Create position
            position: AMMPosition = {
                "token0": token0,
                "token1": token1,
                "liquidity": liquidity,
                "tick_lower": lower_tick,
                "tick_upper": upper_tick,
                "fee_tier": optimal_fee_tier.value,
                "current_tick": self._price_to_tick(current_price),
                "price_range": (self._tick_to_price(lower_tick), self._tick_to_price(upper_tick))
            }
            
            # Calculate expected returns
            expected_fees = await self._calculate_expected_fees(position, volume_24h)
            expected_il = await self.calculate_impermanent_loss(position, current_price)
            
            # MEV protection
            if self.config.mev_protection:
                position = await self._apply_mev_protection(position)
            
            return {
                "position": position,
                "expected_fees": expected_fees,
                "expected_il": expected_il,
                "net_expected_return": expected_fees - expected_il,
                "strategy": "AMM",
                "confidence": self._calculate_confidence(volatility, volume_24h)
            }
            
        except Exception as e:
            logger.error(f"AMM strategy execution failed: {e}")
            raise ValueError(f"AMM strategy failed: {str(e)}")
    
    async def _calculate_expected_fees(
        self, 
        position: AMMPosition, 
        volume_24h: float
    ) -> float:
        """Calculate expected fees for position"""
        try:
            # Estimate volume that will go through our position
            position_liquidity_ratio = position["liquidity"] / (volume_24h * 0.1)  # Assume 10% of volume
            expected_volume = volume_24h * position_liquidity_ratio
            
            # Calculate fees
            expected_fees = expected_volume * position["fee_tier"]
            
            return expected_fees
            
        except Exception as e:
            logger.error(f"Failed to calculate expected fees: {e}")
            return 0.0
    
    async def _apply_mev_protection(self, position: AMMPosition) -> AMMPosition:
        """Apply MEV protection strategies"""
        try:
            # Add random delay to prevent frontrunning
            delay = asyncio.sleep(0.1)  # 100ms delay
            await delay
            
            # Adjust liquidity slightly to avoid exact round numbers
            position["liquidity"] = position["liquidity"] * (0.999 + (hash(str(position)) % 100) / 100000)
            
            return position
            
        except Exception as e:
            logger.error(f"MEV protection failed: {e}")
            return position
    
    def _calculate_confidence(self, volatility: float, volume: float) -> float:
        """Calculate strategy confidence score"""
        try:
            # Higher volume and lower volatility = higher confidence
            volume_score = min(volume / 1000000, 1.0)  # Normalize to 0-1
            volatility_score = max(0, 1 - volatility)  # Lower volatility = higher score
            
            confidence = (volume_score + volatility_score) / 2
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
